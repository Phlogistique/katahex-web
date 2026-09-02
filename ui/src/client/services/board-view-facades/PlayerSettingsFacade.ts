import { AutoOrientationFacade, GameView, themes } from '@playhex/pixi-board';

/**
 * Of the twelve orientations only two put a1 in the top left corner: ORIENTATION_FLAT, which
 * is wide, and this one, the same board turned 30° into a tall shape. So the pair below is the
 * only way to keep a1 in the same corner whichever way the phone is held.
 */
const PORTRAIT_A1_TOP_LEFT = 1;

/**
 * PlayHex applies the logged-in player's board settings here, from stores backed by its server.
 * This app has no account and one look, so it just fixes the settings, keeping the module path
 * so the vendored hexplorer stays a plain copy.
 *
 * The auto orientation is the part that matters: a rhombus laid out corner-up wastes most of a
 * phone screen, and this picks a flatter one in portrait and re-picks it on rotation.
 */
export class PlayerSettingsFacade
{
    constructor(gameView: GameView)
    {
        gameView.setTheme(themes.dark);
        gameView.setDisplayCoords(true);

        new AutoOrientationFacade(gameView, {
            landscape: GameView.ORIENTATION_FLAT,
            portrait: PORTRAIT_A1_TOP_LEFT,
        });
    }
}
