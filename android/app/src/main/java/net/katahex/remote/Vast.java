package net.katahex.remote;

import java.io.IOException;
import java.util.concurrent.TimeUnit;

import org.json.JSONArray;
import org.json.JSONException;
import org.json.JSONObject;

import okhttp3.HttpUrl;
import okhttp3.MediaType;
import okhttp3.OkHttpClient;
import okhttp3.Request;
import okhttp3.RequestBody;
import okhttp3.Response;

/** The few calls of the vast.ai REST API this app makes. Blocking: call off the main thread. */
final class Vast {
    static final String LABEL = "katahex-app";
    static final int PORT = 8765;

    private static final String API = "https://console.vast.ai/api/v0";
    private static final MediaType JSON = MediaType.get("application/json");

    private final OkHttpClient http;
    private final String key;

    Vast(OkHttpClient http, String key) {
        this.http = http.newBuilder().callTimeout(30, TimeUnit.SECONDS).build();
        this.key = key;
    }

    /** Rentable single-GPU machines this image can run on, cheapest first. */
    JSONArray offers(double maxDollarsPerHour) throws IOException {
        try {
            JSONObject q = new JSONObject()
                .put("verified", eq(true))
                .put("rentable", eq(true))
                .put("rented", eq(false))
                .put("num_gpus", eq(1))
                .put("cuda_max_good", new JSONObject().put("gte", 12.2))
                .put("direct_port_count", new JSONObject().put("gte", 1))
                .put("reliability2", new JSONObject().put("gte", 0.95))
                .put("inet_down", new JSONObject().put("gte", 100))
                .put("disk_space", new JSONObject().put("gte", 20))
                .put("gpu_ram", new JSONObject().put("gte", 4000))
                .put("dph_total", new JSONObject().put("lte", maxDollarsPerHour))
                .put("type", "on-demand")
                .put("order", new JSONArray().put(new JSONArray().put("dph_total").put("asc")))
                .put("limit", 20);
            HttpUrl url = HttpUrl.get(API + "/bundles/").newBuilder()
                .addQueryParameter("q", q.toString()).build();
            return call(new Request.Builder().url(url)).getJSONArray("offers");
        } catch (JSONException e) {
            throw new IOException(e);
        }
    }

    /** Rents an offer to run the server image. Returns the new instance's id. */
    long create(long offerId, String image, String token, int idleTimeout, int maxLifetime) throws IOException {
        try {
            String env = "-e KATAHEX_TOKEN=" + token
                + " -e IDLE_TIMEOUT=" + idleTimeout
                + " -e MAX_LIFETIME=" + maxLifetime
                + " -p " + PORT + ":" + PORT;
            JSONObject body = new JSONObject()
                .put("client_id", "me")
                .put("image", image)
                .put("env", env)
                .put("disk", 20)
                .put("label", LABEL)
                .put("runtype", "args")
                .put("args", new JSONArray())
                .put("cancel_unavail", true);
            JSONObject reply = call(new Request.Builder().url(API + "/asks/" + offerId + "/")
                .put(RequestBody.create(body.toString(), JSON)));
            if (!reply.optBoolean("success", false)) {
                throw new IOException("vast refused the rental: " + reply);
            }
            return reply.getLong("new_contract");
        } catch (JSONException e) {
            throw new IOException(e);
        }
    }

    /** The instance, or null if it is gone. */
    JSONObject instance(long id) throws IOException {
        JSONObject reply = call(new Request.Builder().url(API + "/instances/" + id + "/"));
        return reply.optJSONObject("instances");
    }

    JSONArray instances() throws IOException {
        JSONObject reply = call(new Request.Builder().url("https://console.vast.ai/api/v1/instances/"));
        JSONArray list = reply.optJSONArray("instances");
        return list != null ? list : new JSONArray();
    }

    void destroy(long id) throws IOException {
        call(new Request.Builder().url(API + "/instances/" + id + "/").delete());
    }

    private JSONObject call(Request.Builder request) throws IOException {
        request.header("Authorization", "Bearer " + key).header("Accept", "application/json");
        try (Response response = http.newCall(request.build()).execute()) {
            String text = response.body() != null ? response.body().string() : "";
            if (!response.isSuccessful()) {
                throw new IOException("vast " + response.code() + ": " + text.substring(0, Math.min(300, text.length())));
            }
            return text.isEmpty() ? new JSONObject() : new JSONObject(text);
        } catch (JSONException e) {
            throw new IOException(e);
        }
    }

    private static JSONObject eq(Object value) throws JSONException {
        return new JSONObject().put("eq", value);
    }
}
