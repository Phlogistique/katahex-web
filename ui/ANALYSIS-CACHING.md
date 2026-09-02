# Analysis and caching

Hexplorer asks an analyzer for two different things:

- the position on screen: a winrate and the policy overlay;
- every ancestor of the current line, one call each, for the win/lose graph
  (`fillAncestorEvals`, which reads nothing but `whiteWin`).

## One engine, one stop rule

The picker holds five `KatahexAnalyzer`s, differing only in the visit count they stop at: 0, 1,
200, 1000, Infinity. A cell means the same thing at every depth, since `toPolicyGrid` runs
`improvedPolicy` at all of them: a regularized policy, not a share of the visits. At one visit
there are no `moveInfos` and it degenerates to the raw prior, which is what "intuition" is.

The position on screen gets a search that stops at N and reports every half second. Everything
else gets a one-shot at `min(N, 200)`: the graph is not worth more, and auto-play must not act
on a search that has barely started.

Ancestors run in parallel with the search on screen: `numAnalysisThreads = 8`, so it holds one
thread and the graph fills on the other seven. `fillAncestorEvals` submits the whole line at
once because a lone evaluation costs about a second on this gpu and twenty together cost 25ms
each. `unwatch` terminates by id rather than `terminate_all`, so ending the search on screen
does not drop the graph queries in flight.

## The store

`analysisStore` is one localStorage store for every depth, keyed by
`size|colorToPlay|sortedBlack|sortedWhite`, with the net's name in the store name. An entry
holds a winrate and a policy, and the visits behind each. A read asks for a depth and takes
anything deeper; a write only lands when it beats what is stored. The search on screen writes
on every partial report, so a position watched to 50k visits is still worth that next time it
is opened, and is not searched again unless the analyzer stops deeper than it.

The two halves come apart because a search of a position also evaluates its candidates: a
move's winrate is an analysis of the position it leads to. Those are stored too, for candidates
worth at least 200 visits, which reads the lines out of any position sat on for free.

Nothing evicts. `persist` runs when the app goes to the background and when the analyzer
changes, and swallows the quota error rather than dying inside a `visibilitychange` handler.

## The engine's own cache

KataGo's `NNCacheTable`, 2^20 entries on the `NNEvaluator`, keyed by position hash, shared by
all eight analysis threads and never cleared (the app never sends `clear_cache`). It holds
network evaluations, not trees: `command/analysis.cpp` does `setPosition` before every query
and `clearSearch()` after it, so a re-search skips the gpu for nodes it has seen but rebuilds
the tree from nothing. There is no cross-query tree reuse to opt into. It dies with the engine
process, which `ensureSize` restarts on any board size change.

`evalsByNodeId` in `useHexplorer` is a third, weaker layer: node id to `whiteWin`, in memory
only, dropped on analyzer switch, and not part of what `exportAnalysis` writes.
