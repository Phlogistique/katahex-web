# Vendored code

Most of `src` is copied from [PlayHex](https://github.com/playhex/playhex), by Julien Maulny:
its hexplorer analysis board, the PixiJS board renderer and the other packages under it, and
the game, sgf and import code they need. It is **AGPL-3.0-only** (see
`src/packages/pixi-board/LICENSE`), so this app is AGPL-3.0 too. That matters if you ever
distribute it or run it as a network service.

Copied from PlayHex `10ca785a` (2026-11-17). The previous copy was from `4b7b9a5d`
(2026-07-22); between the two PlayHex rewrote the hexplorer into tabs and moved its
publishable code into workspace packages.

It is vendored rather than installed because PlayHex publishes none of it to npm.

The layout mirrors PlayHex's own `src/` (`packages/…`, `shared/…`, `client/…`) so that the
copied files resolve their relative imports without being touched. The packages import each
other by name, `@playhex/pixi-board` and so on, which `vite.config.ts` and `tsconfig.json`
alias to `src/packages`.

## Copied unmodified

    src/packages/pixi-board, move-notation,      PlayHex's workspace packages, without their
      conditional-moves, shading-patterns          tests and cypress setup
    src/shared/game-engine                         board and rules, used to replay a line
    src/shared/sgf
    src/shared/app/hex-game-importer               sgf / hexworld / little golem / raw moves import
    src/shared/app/little-golem                    what the importer needs of it
    src/shared/app/hexplorer.ts, hexplorerStudy.ts, hexplorerSgf.ts,
      boardsizeLimits.ts, mctsSettings.ts, utils.ts
    src/shared/app/en.json                         locales/en.json, plus the keys and wording below
    src/client/vue/components/AppResizableSidebar.vue, AppSidebarNav.vue,
      AppConditionalMoveButton.vue, overlay/ImportGameOverlay.vue
    src/client/vue/hexplorer                       (except what is listed below)

## Written here, at PlayHex's own module paths

These stand in for parts of PlayHex that only make sense with its server behind them. Keeping
the paths is what lets the copied files stay plain copies.

    src/client/services/board-view-facades/PlayerSettingsFacade.ts   no account: fixed settings
    src/client/stores/playerLocalSettingsStore.ts   the one local setting the hexplorer reads
    src/client/vue/hexplorer/composables/useAnalysisEngines.ts   our analyzers, always available
    src/client/vue/hexplorer/analyzers/KatahexAnalyzer.ts        the engine on the Native bridge
    src/client/vue/hexplorer/components/LiveAnalysisButton.vue
    src/shared/app/hexplorerLive.ts                 the two types useHexplorer names, no live sessions
    src/stubs/vue-router.ts, src/stubs/unhead.ts    aliased in vite.config.ts

## Left out

PlayHex's live sessions (streaming an analysis, analyzing together, matches from a position),
its saved and shared studies, and the play tab: all of it is its server. The components and
composables for them are not copied, and the patches below take them out of the files that
are.

## Patches

`src/client/vue/hexplorer/pages/PageHexplorer.vue`: no study, live, play or match; the bottom
bar holds the live analysis button; analysis is on from the start; the sidebar starts closed
on a phone, where it covers the board; the page takes the whole height, there being no PlayHex
header; ctrl+S saves the analysis to a file; the sidebar footer credits the ui, the engine and
the net and links the source, which is where the AGPL's source offer is reachable from.

`src/client/vue/hexplorer/composables/useHexplorer.ts`:

- `updateAnalysis` tells the analyzer which position is on screen (`setDisplayedPosition`),
  and records an evaluation for the graph before checking whether it is stale.
- `fillAncestorEvals` becomes `fillLineEvals`: it covers the whole line the graph draws,
  waits on `whenIdle` before each position, keeps each result as it arrives, asks again for a
  position whose search the screen took back (`SearchAborted`), and reports how far it has got
  (`lineProgress`). There is one engine here rather than a server: a search holds the tree of
  every position it visited until it answers, so a line submitted at once is a line's worth
  of trees resident at once.
- `selectAnalyzer` keeps the evaluations already computed: here every analyzer is one engine
  and one net at a different depth, reading a store that keeps the deepest search of a position.

`src/client/vue/hexplorer/analyzers/AnalyzerInterface.ts`: the two optional methods above,
`setDisplayedPosition` and `whenIdle`.

`src/client/vue/hexplorer/components/HexplorerFileTab.vue`: only the local ways in and out.
The new board size is held to what the engine plays, New asks before throwing a tree away,
and the file loader also takes an sgf, since a phone has files to open where a desktop has a
clipboard to paste from.

`src/client/vue/hexplorer/components/HexplorerTreeTab.vue`: the engine's speed above the
evaluation graph, and how far filling the graph has got under it.

`src/client/vue/hexplorer/components/HexplorerAiTab.vue`: auto-play can be on for both colours
at once, so the engine plays itself; PlayHex locks that out because its analyzers are a server
it would be hammering.

`src/client/vue/hexplorer/components/HexplorerMoveTreePanel.vue`: no ctrl+click mention into a
live session's chat.

`src/client/vue/hexplorer/components/MoveTree.vue`: the node the board is showing is scrolled
into view. A long line runs the tree thousands of pixels past its panel.

`src/client/vue/icons.ts`: only the icons the copied files import. PlayHex's list draws on
icon sets (mdi, tabler, lucide) this app does not install, and the build loads every one it
names.

`src/shared/app/hexworld.ts`: the game fields a hexworld string is made from are a local type
rather than picked from PlayHex's Game model, which would drag in its whole model layer.

`src/shared/app/en.json`: the AI tab and the help tab no longer say analysis is off by default
or that a study can be shared.

## Updating

Re-copy from a PlayHex checkout and re-apply the patches. To find what changed upstream in
the copied files, diff against the commit named above:

    git -C playhex diff 10ca785a HEAD -- src/client/vue/hexplorer src/packages src/shared
