package net.katahex.remote;

import android.app.Activity;
import android.app.AlertDialog;
import android.content.ContentValues;
import android.graphics.Color;
import android.net.Uri;
import android.os.Build;
import android.os.Bundle;
import android.os.Environment;
import android.os.Handler;
import android.os.Looper;
import android.provider.MediaStore;
import android.util.TypedValue;
import android.view.Gravity;
import android.view.ViewGroup;
import android.view.WindowManager;
import android.webkit.JavascriptInterface;
import android.webkit.WebSettings;
import android.webkit.WebView;
import android.widget.LinearLayout;
import android.widget.TextView;
import android.widget.Toast;
import android.window.OnBackInvokedDispatcher;

import java.io.File;
import java.io.FileOutputStream;
import java.io.OutputStream;
import java.nio.charset.StandardCharsets;

import org.json.JSONObject;

/**
 * The hexplorer from ui/ in a WebView, with its engine on a rented GPU.
 *
 * The page talks to its engine through `window.Native`; here that bridge is RemoteEngine,
 * which forwards it over a websocket to the server. Leaving through the back button destroys
 * the server; any other way out leaves it to the server's own dead man's switch.
 */
public class MainActivity extends Activity implements RemoteEngine.Listener {
    private static final long PING_EVERY_MS = 30_000;

    private final Handler main = new Handler(Looper.getMainLooper());
    private WebView web;
    private TextView bar;
    private RemoteEngine engine;
    private boolean resumed;
    private boolean quit;

    private final Runnable pinger = new Runnable() {
        @Override
        public void run() {
            if (resumed) {
                engine.ping();
                main.postDelayed(this, PING_EVERY_MS);
            }
        }
    };

    @Override
    protected void onCreate(Bundle saved) {
        super.onCreate(saved);
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
        engine = new RemoteEngine(getSharedPreferences("server", MODE_PRIVATE), this);

        LinearLayout root = new LinearLayout(this);
        root.setOrientation(LinearLayout.VERTICAL);
        root.setBackgroundColor(Color.rgb(33, 37, 41));

        web = new WebView(this);
        WebSettings settings = web.getSettings();
        settings.setJavaScriptEnabled(true);
        settings.setDomStorageEnabled(true);
        settings.setAllowFileAccess(true);
        web.addJavascriptInterface(new Bridge(), "Native");
        root.addView(web, new LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, 0, 1));

        bar = new TextView(this);
        bar.setTextColor(Color.rgb(173, 181, 189));
        bar.setTextSize(TypedValue.COMPLEX_UNIT_SP, 12);
        bar.setGravity(Gravity.CENTER_VERTICAL);
        bar.setSingleLine(true);
        int pad = (int) (6 * getResources().getDisplayMetrics().density);
        bar.setPadding(pad * 2, pad, pad * 2, pad);
        bar.setText(engine.info().isEmpty() ? "no server yet" : engine.info());
        bar.setOnClickListener(v -> askQuit());
        root.addView(bar, new LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT,
            ViewGroup.LayoutParams.WRAP_CONTENT));

        setContentView(root);
        web.loadUrl("file:///android_asset/index.html");

        if (Build.VERSION.SDK_INT >= 33) {
            getOnBackInvokedDispatcher().registerOnBackInvokedCallback(
                OnBackInvokedDispatcher.PRIORITY_DEFAULT, this::askQuit);
        }
    }

    @Override
    @SuppressWarnings("deprecation")
    public void onBackPressed() {
        askQuit();
    }

    private void askQuit() {
        new AlertDialog.Builder(this)
            .setTitle("Quit?")
            .setMessage("Quitting destroys the GPU server, which stops the bill. "
                + "Leaving the app any other way keeps it for "
                + BuildConfig.IDLE_TIMEOUT / 60 + " minutes, then it destroys itself.")
            .setPositiveButton("Quit and destroy", (d, w) -> quit())
            .setNegativeButton("Stay", null)
            .show();
    }

    private void quit() {
        quit = true;
        bar.setText("destroying the server…");
        engine.quit(() -> main.post(this::finish));
    }

    @Override
    protected void onResume() {
        super.onResume();
        web.onResume();
        resumed = true;
        main.removeCallbacks(pinger);
        main.post(pinger);
    }

    @Override
    protected void onPause() {
        // No pings while the app is not on screen: that is what lets the server's dead man's
        // switch end a session the user walked away from.
        resumed = false;
        main.removeCallbacks(pinger);
        web.onPause();
        super.onPause();
    }

    @Override
    protected void onDestroy() {
        if (isFinishing() && !quit) {
            // Best effort: the process may not live long enough, and the switch covers that.
            engine.quit(() -> {});
        }
        web.destroy();
        super.onDestroy();
    }

    // ---- RemoteEngine.Listener, called from background threads ----------------------------

    @Override
    public void onProgress(String text) {
        call("onEngineProgress", text);
    }

    @Override
    public void onInfo(String text) {
        main.post(() -> bar.setText(text));
    }

    @Override
    public void onEngineLine(String line) {
        call("onEngineLine", line);
    }

    @Override
    public void onEngineLog(String line) {
        call("onEngineLog", line);
    }

    private void call(String function, String argument) {
        String js = "window." + function + " && window." + function + "(" + JSONObject.quote(argument) + ")";
        main.post(() -> web.evaluateJavascript(js, null));
    }

    /** `window.Native`, as ui/src/engine.ts declares it. Called on a WebView binder thread. */
    private final class Bridge {
        @JavascriptInterface
        public void start(int boardSize) {
            engine.start(boardSize);
        }

        @JavascriptInterface
        public void query(String json) {
            engine.query(json);
        }

        @JavascriptInterface
        public boolean running() {
            return engine.running();
        }

        @JavascriptInterface
        public void stop() {
            engine.stop();
        }

        @JavascriptInterface
        public void save(String filename, String mimeType, String content) {
            byte[] bytes = content.getBytes(StandardCharsets.UTF_8);
            try {
                String where;
                if (Build.VERSION.SDK_INT >= 29) {
                    ContentValues values = new ContentValues();
                    values.put(MediaStore.Downloads.DISPLAY_NAME, filename);
                    values.put(MediaStore.Downloads.MIME_TYPE, mimeType);
                    Uri uri = getContentResolver().insert(MediaStore.Downloads.EXTERNAL_CONTENT_URI, values);
                    try (OutputStream out = getContentResolver().openOutputStream(uri)) {
                        out.write(bytes);
                    }
                    where = "Downloads/" + filename;
                } else {
                    File file = new File(getExternalFilesDir(Environment.DIRECTORY_DOWNLOADS), filename);
                    try (OutputStream out = new FileOutputStream(file)) {
                        out.write(bytes);
                    }
                    where = file.getPath();
                }
                toast("saved " + where);
            } catch (Exception e) {
                toast("could not save: " + e.getMessage());
            }
        }
    }

    private void toast(String text) {
        main.post(() -> Toast.makeText(this, text, Toast.LENGTH_LONG).show());
    }
}
