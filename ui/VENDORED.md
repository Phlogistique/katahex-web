# Vendored code

Most of `src` is copied from [PlayHex](https://github.com/playhex/playhex), by Julien Maulny:
its whole hexplorer analysis board, the PixiJS renderer under it, and the game, sgf and import
code they need. It is **AGPL-3.0-only** (see `src/shared/pixi-board/LICENSE`), so this app is
AGPL-3.0 too. That matters if you ever distribute it or run it as a network service.

It is vendored rather than installed because PlayHex publishes none of it to npm.

The layout mirrors PlayHex's own `src/` (`shared/…`, `client/…`) so that the copied files
resolve their relative imports without being touched.

## Copied unmodified

    src/shared/pixi-board                 board renderer, the @playhex/pixi-board workspace
    src/shared/move-notation
    src/shared/resize-observer-debounced
    src/shared/game-engine                board and rules, used to replay a line
    src/shared/sgf
    src/shared/app/hexplorer.ts           the analyzer input/output contract
    src/shared/app/hex-game-importer      sgf / hexworld / little golem / raw moves import
    src/shared/app/en.json                locales/en.json, plus the keys the patches below need
    src/client/vue/components/AppConditionalMoveButton.vue
    src/client/vue/hexplorer                       (except the patches below)

To update, re-copy from a playhex checkout and re-apply the patches.

## Written here, at PlayHex's own module paths

These stand in for parts of PlayHex that only make sense with its server behind them. Keeping
the paths is what lets the copied files stay plain copies.

    src/client/services/board-view-facades/PlayerSettingsFacade.ts   no account: fixed settings
    src/shared/app/hexworld.ts               the two functions hexplorer uses, without the models
    src/client/vue/icons.ts                  only the icons hexplorer uses
    src/stubs/vue-router.ts, src/stubs/unhead.ts   aliased in vite.config.ts

Two files under `hexplorer` are written here rather than copied: the `KatahexAnalyzer`, and the
`LiveAnalysisButton` component.

## Patches

`src/client/vue/hexplorer/pages/PageHexplorer.vue`: the analyzer list comes from
`src/analyzers.ts`, which offers the on-device engine instead of PlayHex's server-side one, and
the bottom menu holds the live analysis button. The load button also takes an sgf, since a phone
has files to open where a desktop has a clipboard to paste from.

`src/client/services/fileDownload.ts`: downloads go through the bridge, since a WebView ignores
an `<a download>` pointing at a blob.

`src/client/vue/hexplorer/analyzers`: PlayHex's own analyzers are dropped, `KatahexAnalyzer`
covers what they do. Two files go with them: `src/client/apiClient.ts`, which stood in for the
server they called, and `hexplorer/services/cachedAnalysis.ts`, which `src/analysisStore.ts`
replaces.

`src/client/vue/hexplorer/analyzers/AnalyzerInterface.ts`: two optional methods.
`setDisplayedPosition` is called with the position now on screen, so an analyzer that keeps
searching can tell it apart from the ancestors it is also asked about. `whenIdle` resolves when
that position needs nothing more, and is what the ancestors wait on.

`src/client/vue/hexplorer/composables/useHexplorer.ts`:

- `updateAnalysis` calls `setDisplayedPosition`.
- `fillAncestorEvals` becomes `fillLineEvals`: it waits on `whenIdle` before each position and
  keeps each result as it arrives. There is one engine here rather than a server: a search holds
  the tree of every position it visited until it answers, in a wasm heap that is never given
  back, so a line submitted at once is a line's worth of trees resident at once. It also covers
  the whole line the graph draws rather than only the moves up to the cursor, and reports how far
  it has got, which the sidebar shows under the graph.
- `setAnalyzer` keeps the evaluations already computed. PlayHex drops them because its
  analyzers are different engines; here they are one engine and one net at different depths,
  reading a store that keeps the deepest search it has of a position.
- The `PlayingGameFacade`s are built with the last move marked, which hexplorer turns off.

`src/client/vue/hexplorer/components/MoveTree.vue`: the node the board is showing is scrolled
into view. A long line runs the tree thousands of pixels past its panel.

`PageHexplorer.vue`, beyond the analyzer list: the board size field is held to what the engine
plays, `New` asks before throwing a tree away, the graph says how far it has got filling, and
the settings checkboxes carry the accessible names their table header gave them. Auto-play can
also be set on both colours at once, so the engine plays itself; `useHexplorer` was already
written for that, and PlayHex only locks it out because its analyzers are a server it would be
hammering.

`PageHexplorer.vue` also carries a credits line in the sidebar footer, naming
where the ui, the engine and the net come from and linking the source. The AGPL
asks for the source of a work served over the network to be reachable, and this
is where it is reachable from.
