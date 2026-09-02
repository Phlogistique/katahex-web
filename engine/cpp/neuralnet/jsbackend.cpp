#ifdef USE_JS_BACKEND

// A backend that does not evaluate the net itself. It writes the inputs into a
// control block in the shared wasm memory, wakes whoever is listening on it, and
// blocks until the results are written back. The listener is a JavaScript worker
// running the net on the GPU through WebGPU.
//
// Everything the engine is really here for -- the board, the input features, the
// search -- stays in C++. Only the forward pass leaves.
//
// The protocol is a ping-pong on one atomic word, `state`:
//
//   engine                                  JavaScript
//   -------------------------------------   ----------------------------------
//   fill inputs, state = REQUEST, notify
//   wait while state == REQUEST             wakes, reads inputs, evaluates
//                                           writes outputs
//                                           state = RESPONSE, notify
//   wakes, reads outputs
//   state = IDLE, notify
//                                           wakes, waits for the next REQUEST
//
// Each net server thread gets its own control block, addressed by its index,
// so two server threads keep two evaluations in flight: while the GPU runs one
// batch, JavaScript packs and submits the other.

#include "../neuralnet/nninterface.h"

#include <emscripten.h>
#include <emscripten/threading.h>

#include <atomic>

#include "../neuralnet/desc.h"
#include "../neuralnet/modelversion.h"
#include "../neuralnet/nneval.h"
#include "../neuralnet/nninputs.h"

using namespace std;

namespace {

enum ControlState : int32_t {
  STATE_IDLE = 0,
  STATE_REQUEST = 1,
  STATE_RESPONSE = 2,
};

// Field offsets, in 32-bit words, of the control block. JavaScript indexes the
// same words by the same names, see `netRunner.ts`.
enum ControlField : int32_t {
  FIELD_STATE = 0,
  FIELD_BATCH_SIZE = 1,
  FIELD_NN_X_LEN = 2,
  FIELD_NN_Y_LEN = 3,
  FIELD_NUM_SPATIAL_FEATURES = 4,
  FIELD_NUM_GLOBAL_FEATURES = 5,
  FIELD_SPATIAL_INPUT = 6,
  FIELD_GLOBAL_INPUT = 7,
  FIELD_POLICY = 8,
  FIELD_POLICY_PASS = 9,
  FIELD_VALUE = 10,
  FIELD_SCORE_VALUE = 11,
  CONTROL_BLOCK_WORDS = 12,
};

constexpr int MAX_SERVER_THREADS = 4;
alignas(16) std::atomic<int32_t> controlBlocks[MAX_SERVER_THREADS][CONTROL_BLOCK_WORDS];

// Waits until the block's state stops being `expected`, then returns what it became.
int32_t waitWhile(std::atomic<int32_t>* block, int32_t expected) {
  for(;;) {
    int32_t seen = block[FIELD_STATE].load(std::memory_order_acquire);
    if(seen != expected)
      return seen;
    emscripten_futex_wait(&block[FIELD_STATE], (uint32_t)expected, INFINITY);
  }
}

void setState(std::atomic<int32_t>* block, int32_t state) {
  block[FIELD_STATE].store(state, std::memory_order_release);
  emscripten_futex_wake(&block[FIELD_STATE], INT_MAX);
}

}  // namespace

// The address JavaScript needs to find a server thread's control block. The
// addresses are fixed for the life of the process, so the runner reads them once.
extern "C" EMSCRIPTEN_KEEPALIVE uint32_t katahexControlBlockAddress(int serverThreadIdx) {
  if(serverThreadIdx < 0 || serverThreadIdx >= MAX_SERVER_THREADS)
    return 0;
  return (uint32_t)(uintptr_t)&controlBlocks[serverThreadIdx][0];
}

//------------------------------------------------------------------------------

struct LoadedModel {
  ModelDesc modelDesc;

  LoadedModel(const string& fileName, const string& expectedSha256) {
    ModelDesc::loadFromFileMaybeGZipped(fileName, modelDesc, expectedSha256);
  }

  LoadedModel() = delete;
  LoadedModel(const LoadedModel&) = delete;
  LoadedModel& operator=(const LoadedModel&) = delete;
};

LoadedModel* NeuralNet::loadModelFile(const string& file, const string& expectedSha256) {
  return new LoadedModel(file, expectedSha256);
}

void NeuralNet::freeLoadedModel(LoadedModel* loadedModel) {
  delete loadedModel;
}

string NeuralNet::getModelName(const LoadedModel* loadedModel) {
  return loadedModel->modelDesc.name;
}

int NeuralNet::getModelVersion(const LoadedModel* loadedModel) {
  return loadedModel->modelDesc.version;
}

Rules NeuralNet::getSupportedRules(const LoadedModel* loadedModel, const Rules& desiredRules, bool& supported) {
  return loadedModel->modelDesc.getSupportedRules(desiredRules, supported);
}

void NeuralNet::globalInitialize() {
}

void NeuralNet::globalCleanup() {
}

void NeuralNet::printDevices() {
}

//------------------------------------------------------------------------------

struct ComputeContext {
  int nnXLen;
  int nnYLen;
};

ComputeContext* NeuralNet::createComputeContext(
  const std::vector<int>& gpuIdxs,
  Logger* logger,
  int nnXLen,
  int nnYLen,
  const string& openCLTunerFile,
  const string& homeDataDirOverride,
  bool openCLReTunePerBoardSize,
  enabled_t useFP16Mode,
  enabled_t useNHWCMode,
  const LoadedModel* loadedModel
) {
  (void)gpuIdxs;
  (void)logger;
  (void)openCLTunerFile;
  (void)homeDataDirOverride;
  (void)openCLReTunePerBoardSize;
  (void)useFP16Mode;
  (void)loadedModel;

  if(useNHWCMode == enabled_t::False)
    throw StringError("JS backend: useNHWC = false not supported");

  ComputeContext* context = new ComputeContext();
  context->nnXLen = nnXLen;
  context->nnYLen = nnYLen;
  return context;
}

void NeuralNet::freeComputeContext(ComputeContext* computeContext) {
  delete computeContext;
}

//------------------------------------------------------------------------------

struct ComputeHandle {
  const ComputeContext* context;
  const ModelDesc* modelDesc;
  int maxBatchSize;
  std::atomic<int32_t>* controlBlock;

  vector<float> policy;
  vector<float> policyPass;
  vector<float> value;
  vector<float> scoreValue;

  ComputeHandle(const ComputeContext* ctx, const LoadedModel& loadedModel, int maxBatchSz, int serverThreadIdx)
    : context(ctx), modelDesc(&loadedModel.modelDesc), maxBatchSize(maxBatchSz),
      controlBlock(controlBlocks[serverThreadIdx]) {
    const ModelDesc& m = loadedModel.modelDesc;
    policy = vector<float>((size_t)maxBatchSize * ctx->nnXLen * ctx->nnYLen);
    policyPass = vector<float>((size_t)maxBatchSize);
    value = vector<float>((size_t)maxBatchSize * m.numValueChannels);
    scoreValue = vector<float>((size_t)maxBatchSize * m.numScoreValueChannels);
  }

  ComputeHandle() = delete;
  ComputeHandle(const ComputeHandle&) = delete;
  ComputeHandle& operator=(const ComputeHandle&) = delete;
};

ComputeHandle* NeuralNet::createComputeHandle(
  ComputeContext* context,
  const LoadedModel* loadedModel,
  Logger* logger,
  int maxBatchSize,
  bool requireExactNNLen,
  bool inputsUseNHWC,
  int gpuIdxForThisThread,
  int serverThreadIdx
) {
  (void)requireExactNNLen;
  (void)gpuIdxForThisThread;

  if(!inputsUseNHWC)
    throw StringError("JS backend: inputsUseNHWC = false not supported");
  if(serverThreadIdx < 0 || serverThreadIdx >= MAX_SERVER_THREADS)
    throw StringError("JS backend: at most " + Global::intToString(MAX_SERVER_THREADS) + " server threads");
  if(logger != NULL)
    logger->write(
      "JS backend thread " + Global::intToString(serverThreadIdx) + ": Model version " +
      Global::intToString(loadedModel->modelDesc.version));

  return new ComputeHandle(context, *loadedModel, maxBatchSize, serverThreadIdx);
}

void NeuralNet::freeComputeHandle(ComputeHandle* computeHandle) {
  delete computeHandle;
}

bool NeuralNet::isUsingFP16(const ComputeHandle* computeHandle) {
  (void)computeHandle;
  return false;
}

//------------------------------------------------------------------------------

struct InputBuffers {
  int maxBatchSize;
  size_t singleInputElts;
  size_t singleInputGlobalElts;
  size_t singlePolicyResultElts;

  vector<float> spatialInput;
  vector<float> globalInput;

  InputBuffers(const LoadedModel* loadedModel, int maxBatchSz, int nnXLen, int nnYLen) {
    const ModelDesc& m = loadedModel->modelDesc;
    maxBatchSize = maxBatchSz;
    singleInputElts = (size_t)m.numInputChannels * nnXLen * nnYLen;
    singleInputGlobalElts = (size_t)m.numInputGlobalChannels;
    singlePolicyResultElts = (size_t)nnXLen * nnYLen;

    spatialInput = vector<float>(singleInputElts * maxBatchSize);
    globalInput = vector<float>(singleInputGlobalElts * maxBatchSize);
  }

  InputBuffers() = delete;
  InputBuffers(const InputBuffers&) = delete;
  InputBuffers& operator=(const InputBuffers&) = delete;
};

InputBuffers* NeuralNet::createInputBuffers(const LoadedModel* loadedModel, int maxBatchSize, int nnXLen, int nnYLen) {
  return new InputBuffers(loadedModel, maxBatchSize, nnXLen, nnYLen);
}

void NeuralNet::freeInputBuffers(InputBuffers* inputBuffers) {
  delete inputBuffers;
}

//------------------------------------------------------------------------------

void NeuralNet::getOutput(
  ComputeHandle* computeHandle,
  InputBuffers* inputBuffers,
  int numBatchEltsFilled,
  NNResultBuf** inputBufs,
  vector<NNOutput*>& outputs
) {
  assert(numBatchEltsFilled > 0 && numBatchEltsFilled <= inputBuffers->maxBatchSize);
  const int batchSize = numBatchEltsFilled;
  const int nnXLen = computeHandle->context->nnXLen;
  const int nnYLen = computeHandle->context->nnYLen;
  const ModelDesc& model = *computeHandle->modelDesc;
  const int version = model.version;
  const int numSpatialFeatures = NNModelVersion::getNumSpatialFeatures(version);
  const int numGlobalFeatures = NNModelVersion::getNumGlobalFeatures(version);

  for(int nIdx = 0; nIdx < batchSize; nIdx++) {
    float* rowSpatialInput = inputBuffers->spatialInput.data() + inputBuffers->singleInputElts * nIdx;
    float* rowGlobalInput = inputBuffers->globalInput.data() + inputBuffers->singleInputGlobalElts * nIdx;
    const float* rowGlobal = inputBufs[nIdx]->rowGlobal;
    const float* rowSpatial = inputBufs[nIdx]->rowSpatial;
    std::copy(rowGlobal, rowGlobal + numGlobalFeatures, rowGlobalInput);
    SymmetryHelpers::copyInputsWithSymmetry(
      rowSpatial, rowSpatialInput, 1, nnYLen, nnXLen, numSpatialFeatures, true, inputBufs[nIdx]->symmetry);
  }

  {
    std::atomic<int32_t>* block = computeHandle->controlBlock;
    block[FIELD_BATCH_SIZE].store(batchSize);
    block[FIELD_NN_X_LEN].store(nnXLen);
    block[FIELD_NN_Y_LEN].store(nnYLen);
    block[FIELD_NUM_SPATIAL_FEATURES].store(numSpatialFeatures);
    block[FIELD_NUM_GLOBAL_FEATURES].store(numGlobalFeatures);
    block[FIELD_SPATIAL_INPUT].store((int32_t)(uintptr_t)inputBuffers->spatialInput.data());
    block[FIELD_GLOBAL_INPUT].store((int32_t)(uintptr_t)inputBuffers->globalInput.data());
    block[FIELD_POLICY].store((int32_t)(uintptr_t)computeHandle->policy.data());
    block[FIELD_POLICY_PASS].store((int32_t)(uintptr_t)computeHandle->policyPass.data());
    block[FIELD_VALUE].store((int32_t)(uintptr_t)computeHandle->value.data());
    block[FIELD_SCORE_VALUE].store((int32_t)(uintptr_t)computeHandle->scoreValue.data());

    setState(block, STATE_REQUEST);
    int32_t seen = waitWhile(block, STATE_REQUEST);
    if(seen != STATE_RESPONSE)
      throw StringError("JS backend: unexpected control state " + Global::intToString(seen));
    setState(block, STATE_IDLE);
  }

  assert(outputs.size() == (size_t)batchSize);
  const int numValueChannels = model.numValueChannels;
  const int numScoreValueChannels = model.numScoreValueChannels;
  assert(numValueChannels == 3);

  for(int row = 0; row < batchSize; row++) {
    NNOutput* output = outputs[row];
    assert(output->nnXLen == nnXLen && output->nnYLen == nnYLen);

    const float* policySrcBuf = computeHandle->policy.data() + row * inputBuffers->singlePolicyResultElts;
    SymmetryHelpers::copyOutputsWithSymmetry(
      policySrcBuf, output->policyProbs, 1, nnYLen, nnXLen, inputBufs[row]->symmetry);
    output->policyProbs[inputBuffers->singlePolicyResultElts] = computeHandle->policyPass[row];

    output->whiteWinProb = computeHandle->value[row * numValueChannels];
    output->whiteLossProb = computeHandle->value[row * numValueChannels + 1];
    output->whiteNoResultProb = computeHandle->value[row * numValueChannels + 2];

    // Only versions 9 and up are reachable here: the hex nets are version 11, and
    // anything older lacks these two heads.
    assert(version >= 9 && numScoreValueChannels == 6);
    output->varTimeLeft = computeHandle->scoreValue[row * numScoreValueChannels + 3];
    output->shorttermWinlossError = computeHandle->scoreValue[row * numScoreValueChannels + 4];
  }
}

//------------------------------------------------------------------------------
// Layer-by-layer tests have no meaning for a backend that owns no layers.

bool NeuralNet::testEvaluateConv(
  const ConvLayerDesc*, int, int, int, bool, bool, const vector<float>&, vector<float>&) {
  return false;
}

bool NeuralNet::testEvaluateBatchNorm(
  const BatchNormLayerDesc*, int, int, int, bool, bool, const vector<float>&, const vector<float>&, vector<float>&) {
  return false;
}

bool NeuralNet::testEvaluateResidualBlock(
  const ResidualBlockDesc*, int, int, int, bool, bool, const vector<float>&, const vector<float>&, vector<float>&) {
  return false;
}

bool NeuralNet::testEvaluateGlobalPoolingResidualBlock(
  const GlobalPoolingResidualBlockDesc*, int, int, int, bool, bool, const vector<float>&, const vector<float>&,
  vector<float>&) {
  return false;
}

#endif  // USE_JS_BACKEND
