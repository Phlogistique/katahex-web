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
    src/shared/app/en.json                locales/en.json
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

`src/client/vue/hexplorer/analyzers/AnalyzerInterface.ts`: an optional `setDisplayedPosition`,
called with the position now on screen. An analyzer that keeps searching needs to tell it apart
from the ancestors it is also asked about.

`src/client/vue/hexplorer/composables/useHexplorer.ts`:

- `updateAnalysis` calls `setDisplayedPosition`.
- `fillAncestorEvals` submits the line's positions together rather than one after the other. On
  the phone gpu a lone evaluation costs about a second and twenty at once cost 25ms each, so
  reviewing a whole line is only bearable batched.
- The `PlayingGameFacade`s are built with the last move marked, which hexplorer turns off.

`PageHexplorer.vue` also carries a credits line in the sidebar footer, naming
where the ui, the engine and the net come from and linking the source. The AGPL
asks for the source of a work served over the network to be reachable, and this
is where it is reachable from.
