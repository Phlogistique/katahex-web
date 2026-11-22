package net.katahex.remote;

import android.app.Activity;
import android.app.AlertDialog;
import android.content.SharedPreferences;
import android.graphics.Typeface;
import android.os.Bundle;
import android.os.Handler;
import android.os.Looper;
import android.text.InputType;
import android.widget.Button;
import android.widget.EditText;
import android.widget.LinearLayout;
import android.widget.ScrollView;
import android.widget.TextView;

import java.io.IOException;
import java.security.SecureRandom;
import java.util.HashSet;
import java.util.Locale;
import java.util.Set;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
import java.util.regex.Pattern;

import org.json.JSONArray;
import org.json.JSONException;
import org.json.JSONObject;

import okhttp3.OkHttpClient;
import okhttp3.Request;
import okhttp3.RequestBody;
import okhttp3.Response;

/**
 * Rents a GPU on vast.ai to run PlayHex AI workers (the image in worker/), and shows how they do.
 *
 * While the app is on screen it pings the machine every 30 s. Once it is left, the machine's
 * dead man's switch destroys it IDLE_TIMEOUT later; "Stop" destroys it at once.
 */
public class WorkerActivity extends Activity {
    private static final String LABEL = "playhex-worker";
    private static final long PING_EVERY_MS = 30_000;
    /** What a host's status says when its docker, rather than our image, has failed. */
    private static final Pattern HOST_FAILURE = Pattern.compile(
        "(?i)error|unknown flag|usage:|failed|cannot|no such|not found|denied");

    private final Handler main = new Handler(Looper.getMainLooper());
    private final ExecutorService background = Executors.newSingleThreadExecutor();
    private final OkHttpClient http = new OkHttpClient.Builder().callTimeout(15, TimeUnit.SECONDS).build();
    /** Machines that failed to start the image, not to be rented again by this run of the app. */
    private final Set<Long> broken = new HashSet<>();
    private SharedPreferences prefs;
    private TextView status;
    private TextView log;
    private Button action;
    private boolean resumed;

    private final Runnable pinger = new Runnable() {
        @Override
        public void run() {
            if (resumed) {
                background.execute(WorkerActivity.this::ping);
                main.postDelayed(this, PING_EVERY_MS);
            }
        }
    };

    @Override
    protected void onCreate(Bundle saved) {
        super.onCreate(saved);
        prefs = getSharedPreferences("worker", MODE_PRIVATE);
        int pad = (int) (16 * getResources().getDisplayMetrics().density);

        LinearLayout root = new LinearLayout(this);
        root.setOrientation(LinearLayout.VERTICAL);
        root.setPadding(pad, pad, pad, pad);
        root.setFitsSystemWindows(true);
        status = new TextView(this);
        status.setTextSize(16);
        root.addView(status);

        LinearLayout buttons = new LinearLayout(this);
        action = new Button(this);
        action.setOnClickListener(v -> {
            if (prefs.getLong("instance", 0) == 0) {
                background.execute(this::start);
            } else {
                background.execute(this::stop);
            }
        });
        buttons.addView(action);
        Button keys = new Button(this);
        keys.setText("Keys");
        keys.setOnClickListener(v -> askKeys());
        buttons.addView(keys);
        root.addView(buttons);

        log = new TextView(this);
        log.setTypeface(Typeface.MONOSPACE);
        log.setTextSize(10);
        ScrollView scroll = new ScrollView(this);
        scroll.addView(log);
        root.addView(scroll, new LinearLayout.LayoutParams(LinearLayout.LayoutParams.MATCH_PARENT, 0, 1));
        setContentView(root);

        show(prefs.getLong("instance", 0) == 0 ? "No worker running" : prefs.getString("info", ""), null);
        if (vastKey().isEmpty() || workerKey().isEmpty()) {
            askKeys();
        }
    }

    @Override
    protected void onResume() {
        super.onResume();
        resumed = true;
        main.removeCallbacks(pinger);
        main.post(pinger);
    }

    @Override
    protected void onPause() {
        // No pings while the app is not on screen: that is what lets the machine's dead man's
        // switch end a session the user walked away from.
        resumed = false;
        main.removeCallbacks(pinger);
        super.onPause();
    }

    // ---- on the background thread -----------------------------------------------------

    private void start() {
        if (prefs.getLong("instance", 0) != 0) {
            return; // tapped twice
        }
        if (vastKey().isEmpty() || workerKey().isEmpty()) {
            show("Set both keys first", null);
            return;
        }
        try {
            // Machines this app rented and lost track of, which would otherwise run up a bill.
            JSONArray all = vast().instances();
            for (int i = 0; i < all.length(); i++) {
                JSONObject instance = all.getJSONObject(i);
                if (LABEL.equals(instance.optString("label"))) {
                    vast().destroy(instance.getLong("id"));
                }
            }
            for (int attempt = 0; attempt < 3; attempt++) {
                long id = rent();
                if (waitUp(id)) {
                    ping();
                    return;
                }
            }
            show("Three machines in a row failed to start, giving up", null);
        } catch (IOException | JSONException e) {
            show("Could not start: " + e.getMessage(), null);
        }
    }

    private long rent() throws IOException, JSONException {
        show("Looking for the cheapest GPU on vast.ai", null);
        JSONArray offers = vast().offers(BuildConfig.MAX_DOLLARS_PER_HOUR);
        IOException last = new IOException("no GPU offer under $" + BuildConfig.MAX_DOLLARS_PER_HOUR + "/h");
        // The cheapest are often taken between the search and the rental: walk down the list.
        for (int i = 0, tried = 0; i < offers.length() && tried < 5; i++) {
            JSONObject offer = offers.getJSONObject(i);
            String gpu = offer.optString("gpu_name", "GPU");
            if (gpu.startsWith("GTX") || broken.contains(offer.optLong("machine_id"))) {
                continue;
            }
            tried++;
            String token = newToken();
            String info = String.format(Locale.US, "%s · $%.3f/h", gpu, offer.optDouble("dph_total"));
            show("Renting a " + info, null);
            try {
                long id = vast().create(offer.getLong("id"), BuildConfig.IMAGE, LABEL, "-e WORKER_TOKEN=" + token
                    + " -e AI_WORKER_KEY=" + workerKey() + " -e HEX_URL=" + BuildConfig.HEX_URL
                    + " -e IDLE_TIMEOUT=" + BuildConfig.IDLE_TIMEOUT + " -e FIRST_CONNECT_TIMEOUT=" + BuildConfig.IDLE_TIMEOUT
                    + " -e MAX_LIFETIME=" + BuildConfig.MAX_LIFETIME);
                prefs.edit().putLong("instance", id).putString("token", token).putString("info", info + " · #" + id)
                    .remove("address").apply();
                return id;
            } catch (IOException e) {
                last = e;
            }
        }
        throw last;
    }

    /** Waits for the machine to answer /health. False if it failed and was destroyed. */
    private boolean waitUp(long id) throws IOException {
        long since = System.currentTimeMillis();
        while (true) {
            JSONObject instance = vast().instance(id);
            if (instance == null) {
                forget("The machine is gone");
                return false;
            }
            String state = instance.optString("actual_status", "starting");
            String message = instance.optString("status_msg", "").trim();
            long waited = (System.currentTimeMillis() - since) / 1000;
            boolean failed = !"running".equals(state) && HOST_FAILURE.matcher(message).find();
            // A pull takes two or three minutes; past ten something is wrong with the machine.
            if (failed || waited > 600 || "exited".equals(state) || "offline".equals(state)) {
                broken.add(instance.optLong("machine_id"));
                vast().destroy(id);
                forget("That machine failed (" + message.split("\n")[0] + "), trying another");
                return false;
            }
            String address = address(instance);
            if ("running".equals(state) && address != null && get("http://" + address + "/health") != null) {
                prefs.edit().putString("address", address).apply();
                return true;
            }
            show(String.format(Locale.US, "%s\nMachine %s (%ds)%s", prefs.getString("info", ""),
                "null".equals(state) ? "starting" : state, waited,
                message.isEmpty() || "null".equals(message) ? "" : ": " + message.substring(0, Math.min(80, message.length()))), null);
            try {
                Thread.sleep(4000);
            } catch (InterruptedException e) {
                return false;
            }
        }
    }

    private void ping() {
        long id = prefs.getLong("instance", 0);
        String address = prefs.getString("address", null);
        if (id == 0) {
            return;
        }
        try {
            String reply = address == null ? null : get("http://" + address + "/ping?token=" + prefs.getString("token", ""));
            if (reply != null) {
                JSONObject s = new JSONObject(reply);
                int left = s.getInt("left");
                JSONArray lines = s.getJSONArray("log");
                StringBuilder text = new StringBuilder();
                for (int i = lines.length() - 1; i >= 0; i--) {
                    text.append(lines.getString(i)).append('\n');
                }
                show(String.format(Locale.US, "%s\n%d workers · %d jobs done · switch in %d:%02d",
                    prefs.getString("info", ""), s.getInt("workers"), s.getInt("done"), left / 60, left % 60),
                    text.toString());
                return;
            }
            // No answer: the machine may be gone, or still coming up after the app was killed.
            if (vast().instance(id) == null) {
                forget("The machine is gone: its dead man's switch fired, or it died");
            } else if (waitUp(id)) {
                ping();
            }
        } catch (IOException | JSONException e) {
            show(prefs.getString("info", "") + "\nCannot reach the machine: " + e.getMessage(), null);
        }
    }

    private void stop() {
        long id = prefs.getLong("instance", 0);
        String address = prefs.getString("address", null);
        if (id == 0) {
            return; // tapped twice
        }
        show("Destroying the machine", null);
        if (address != null) {
            try (Response r = http.newCall(new Request.Builder()
                    .url("http://" + address + "/bye?token=" + prefs.getString("token", ""))
                    .post(RequestBody.create(new byte[0])).build()).execute()) {
                // Lets the workers give back their jobs before the machine goes.
                Thread.sleep(3000);
            } catch (IOException | InterruptedException e) {
                // vast's own destroy below does it anyway.
            }
        }
        try {
            vast().destroy(id);
            forget("No worker running");
        } catch (IOException e) {
            show("Could not destroy the machine, its switch will: " + e.getMessage(), null);
        }
    }

    // ---- small things -------------------------------------------------------------------

    private String get(String url) {
        try (Response r = http.newCall(new Request.Builder().url(url).build()).execute()) {
            return r.isSuccessful() && r.body() != null ? r.body().string() : null;
        } catch (IOException e) {
            return null;
        }
    }

    private static String address(JSONObject instance) {
        String ip = instance.optString("public_ipaddr", "").trim();
        JSONObject ports = instance.optJSONObject("ports");
        JSONArray mapped = ports != null ? ports.optJSONArray(Vast.PORT + "/tcp") : null;
        if (ip.isEmpty() || mapped == null || mapped.length() == 0) {
            return null;
        }
        return ip + ":" + mapped.optJSONObject(0).optString("HostPort");
    }

    private void forget(String text) {
        prefs.edit().remove("instance").remove("token").remove("address").remove("info").apply();
        show(text, null);
    }

    /** Status, and the workers' log when there is one. Any thread. */
    private void show(String text, String lines) {
        boolean running = prefs.getLong("instance", 0) != 0;
        main.post(() -> {
            status.setText(text);
            action.setText(running ? "Stop and destroy" : "Start a worker");
            if (lines != null) {
                log.setText(lines);
            }
        });
    }

    private Vast vast() {
        return new Vast(http, vastKey());
    }

    private String vastKey() {
        return prefs.getString("vast_key", "");
    }

    private String workerKey() {
        return prefs.getString("worker_key", "");
    }

    private void askKeys() {
        LinearLayout box = new LinearLayout(this);
        box.setOrientation(LinearLayout.VERTICAL);
        int pad = (int) (20 * getResources().getDisplayMetrics().density);
        box.setPadding(pad, 0, pad, 0);
        EditText vast = field("vast.ai API key", vastKey());
        EditText worker = field("PlayHex AI worker key", workerKey());
        box.addView(vast);
        box.addView(worker);
        new AlertDialog.Builder(this)
            .setTitle("Keys")
            .setMessage("The vast.ai key rents the GPU on your account (cloud.vast.ai, Account, Keys) and stays "
                + "on this phone. The PlayHex key is in your PlayHex settings, AI workers; ask the admins for one.")
            .setView(box)
            .setPositiveButton("Save", (d, w) -> prefs.edit()
                .putString("vast_key", vast.getText().toString().trim())
                .putString("worker_key", worker.getText().toString().trim()).apply())
            .setNegativeButton("Cancel", null)
            .show();
    }

    private EditText field(String hint, String value) {
        EditText field = new EditText(this);
        field.setSingleLine(true);
        field.setInputType(InputType.TYPE_CLASS_TEXT | InputType.TYPE_TEXT_VARIATION_VISIBLE_PASSWORD);
        field.setHint(hint);
        field.setText(value);
        return field;
    }

    private static String newToken() {
        byte[] bytes = new byte[24];
        new SecureRandom().nextBytes(bytes);
        StringBuilder s = new StringBuilder();
        for (byte b : bytes) {
            s.append(String.format(Locale.US, "%02x", b));
        }
        return s.toString();
    }
}
