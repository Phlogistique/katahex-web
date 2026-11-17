package net.katahex.remote;

import android.content.SharedPreferences;

import java.io.IOException;
import java.security.SecureRandom;
import java.util.Locale;
import java.util.concurrent.Executors;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.TimeUnit;

import org.json.JSONArray;
import org.json.JSONException;
import org.json.JSONObject;

import okhttp3.OkHttpClient;
import okhttp3.Request;
import okhttp3.Response;
import okhttp3.WebSocket;
import okhttp3.WebSocketListener;

/**
 * The engine, on a GPU rented for as long as the app is in use.
 *
 * Brings a vast.ai instance up (or finds the one it left running), connects to the server on
 * it, and carries the page's engine bridge over the websocket. Whatever goes wrong -- the
 * machine dies, the dead man's switch fires while the phone was put away, the network drops --
 * it starts over from "is my instance still there", renting a new one if not.
 */
final class RemoteEngine {
    interface Listener {
        /** Progress while the engine is not up, which the page shows in place of the board's status. */
        void onProgress(String text);

        /** One line about the server, for the status bar under the board. */
        void onInfo(String text);

        void onEngineLine(String line);

        void onEngineLog(String line);
    }

    private static final String PREF_INSTANCE = "instance";
    private static final String PREF_TOKEN = "token";
    private static final String PREF_INFO = "info";
    /** Machines that failed to start our container, never to be rented again. */
    private static final String PREF_BROKEN = "broken";
    /** What a host's status says when its docker, rather than our image, has failed. */
    private static final java.util.regex.Pattern HOST_FAILURE = java.util.regex.Pattern.compile(
        "(?i)error|unknown flag|usage:|failed|cannot|no such|not found|denied");

    private final OkHttpClient http = new OkHttpClient.Builder()
        .connectTimeout(15, TimeUnit.SECONDS)
        .readTimeout(0, TimeUnit.SECONDS)
        .pingInterval(20, TimeUnit.SECONDS)
        .build();
    private static final String PREF_KEY = "vast_key";
    private volatile Vast vast;
    private final ScheduledExecutorService worker = Executors.newSingleThreadScheduledExecutor();
    private final SharedPreferences prefs;
    private final Listener listener;

    // Touched from the worker, the websocket's thread and the page's bridge thread.
    private volatile WebSocket socket;
    private volatile boolean connected;
    private volatile boolean quitting;
    private volatile boolean bringingUp;
    private volatile int wantSize;
    private volatile String info = "";

    RemoteEngine(SharedPreferences prefs, Listener listener) {
        this.prefs = prefs;
        this.listener = listener;
        info = prefs.getString(PREF_INFO, "");
        vast = new Vast(http, key());
    }

    /** The key built into the apk, or else the one the user typed in. */
    String key() {
        return BuildConfig.VAST_API_KEY.isEmpty() ? prefs.getString(PREF_KEY, "") : BuildConfig.VAST_API_KEY;
    }

    boolean keyBuiltIn() {
        return !BuildConfig.VAST_API_KEY.isEmpty();
    }

    void setKey(String key) {
        prefs.edit().putString(PREF_KEY, key.trim()).apply();
        vast = new Vast(http, key());
        ensureUp();
    }

    // ---- the page's bridge --------------------------------------------------------------

    void start(int size) {
        wantSize = size;
        if (connected) {
            send(message("start").put("size", size));
        } else {
            ensureUp();
        }
    }

    /** Dropped while disconnected: on reconnect the page is told the engine is ready, and resends. */
    void query(String json) {
        if (connected) {
            send(message("q").put("q", json));
        }
    }

    boolean running() {
        return connected;
    }

    void stop() {
        if (connected) {
            send(message("stop"));
        }
    }

    // ---- life ---------------------------------------------------------------------------

    /** A sign of life for the dead man's switch; only sent while the app is on screen. */
    void ping() {
        if (connected) {
            send(message("ping"));
        } else {
            ensureUp();
        }
    }

    /** Destroys the server and calls back once vast has said so, or has failed to. */
    void quit(Runnable done) {
        quitting = true;
        if (connected) {
            send(message("bye"));
        }
        WebSocket ws = socket;
        if (ws != null) {
            ws.close(1000, "bye");
        }
        worker.execute(() -> {
            long id = prefs.getLong(PREF_INSTANCE, 0);
            if (id != 0) {
                try {
                    vast.destroy(id);
                } catch (IOException e) {
                    // The server heard "bye" or will time out on its own.
                }
            }
            prefs.edit().remove(PREF_INSTANCE).remove(PREF_TOKEN).remove(PREF_INFO).apply();
            done.run();
        });
    }

    String info() {
        return info;
    }

    // ---- bringing a server up -----------------------------------------------------------

    void ensureUp() {
        if (quitting || connected || bringingUp) {
            return;
        }
        bringingUp = true;
        worker.execute(this::bringUp);
    }

    private void bringUp() {
        try {
            if (key().isEmpty()) {
                progress("enter your vast.ai API key: tap the bar at the bottom");
                bringingUp = false;
                return;
            }
            long id = prefs.getLong(PREF_INSTANCE, 0);
            String token = prefs.getString(PREF_TOKEN, null);
            JSONObject instance = id != 0 && token != null ? vast.instance(id) : null;

            if (instance == null || dead(instance)) {
                if (instance != null) {
                    vast.destroy(id);
                }
                destroyStrays();
                token = newToken();
                id = rent(token);
            } else {
                progress("found the server left running, reconnecting");
            }

            String address = waitForServer(id);
            if (address == null) {
                return;
            }
            connect(address, token);
        } catch (Exception e) {
            progress("server: " + e.getMessage() + " -- retrying");
            bringingUp = false;
            if (!quitting) {
                worker.schedule(this::ensureUp, 10, TimeUnit.SECONDS);
            }
            return;
        }
    }

    /** Instances this app rented and lost track of, which would otherwise run up a bill. */
    private void destroyStrays() throws IOException {
        JSONArray all = vast.instances();
        for (int i = 0; i < all.length(); i++) {
            JSONObject instance = all.optJSONObject(i);
            if (instance != null && Vast.LABEL.equals(instance.optString("label"))) {
                progress("destroying a server left behind: " + instance.optLong("id"));
                vast.destroy(instance.optLong("id"));
            }
        }
    }

    private long rent(String token) throws IOException, JSONException {
        progress("looking for the cheapest GPU on vast.ai");
        JSONArray offers = vast.offers(BuildConfig.MAX_DOLLARS_PER_HOUR);
        if (offers.length() == 0) {
            throw new IOException("no GPU offer under $" + BuildConfig.MAX_DOLLARS_PER_HOUR + "/h");
        }
        IOException last = new IOException("every offer was a GTX");
        int tried = 0;
        // The cheapest are often taken between the search and the rental: walk down the list.
        for (int i = 0; i < offers.length() && tried < 5; i++) {
            JSONObject offer = offers.getJSONObject(i);
            String gpu = offer.optString("gpu_name", "GPU");
            // A GTX 16xx is Turing without the tensor cores, so as slow as the Pascal it replaced.
            if (gpu.startsWith("GTX") || brokenMachines().contains(String.valueOf(offer.optLong("machine_id")))) {
                continue;
            }
            tried++;
            double price = offer.optDouble("dph_total", 0);
            try {
                progress(String.format(Locale.US, "renting a %s at $%.3f/h", gpu, price));
                long id = vast.create(offer.getLong("id"), BuildConfig.IMAGE, token,
                    BuildConfig.IDLE_TIMEOUT, BuildConfig.MAX_LIFETIME);
                setInfo(String.format(Locale.US, "%s · $%.3f/h · #%d", gpu, price, id));
                prefs.edit().putLong(PREF_INSTANCE, id).putString(PREF_TOKEN, token)
                    .putString(PREF_INFO, info).apply();
                return id;
            } catch (IOException e) {
                last = e;
            }
        }
        throw last;
    }

    private java.util.Set<String> brokenMachines() {
        return prefs.getStringSet(PREF_BROKEN, java.util.Collections.emptySet());
    }

    private static boolean dead(JSONObject instance) {
        String status = instance.optString("actual_status", "");
        return status.equals("exited") || status.equals("offline");
    }

    /** Waits for the instance to run and its server to answer. Returns host:port, or null if quitting. */
    private String waitForServer(long id) throws IOException, InterruptedException {
        long since = System.currentTimeMillis();
        long runningSince = 0;
        while (!quitting) {
            JSONObject instance = vast.instance(id);
            if (instance == null) {
                prefs.edit().remove(PREF_INSTANCE).apply();
                throw new IOException("the server is gone");
            }
            if (dead(instance)) {
                vast.destroy(id);
                prefs.edit().remove(PREF_INSTANCE).apply();
                throw new IOException("the machine stopped (" + instance.optString("status_msg") + ")");
            }
            String status = instance.optString("actual_status", "starting");
            String message = instance.optString("status_msg", "");
            if (!"running".equals(status) && HOST_FAILURE.matcher(message).find()) {
                vast.destroy(id);
                java.util.Set<String> broken = new java.util.HashSet<>(brokenMachines());
                broken.add(String.valueOf(instance.optLong("machine_id")));
                prefs.edit().remove(PREF_INSTANCE).putStringSet(PREF_BROKEN, broken).apply();
                String first = message.trim().split("\n")[0];
                throw new IOException("that machine is broken (" + first.substring(0, Math.min(60, first.length()))
                    + "), renting another");
            }
            String address = address(instance);
            long waited = (System.currentTimeMillis() - since) / 1000;
            boolean unreachable = false;
            if ("running".equals(status) && address != null) {
                if (healthy(address)) {
                    return address;
                }
                // Running, mapped, and not answering: the server inside may still be starting,
                // but after a minute it is more likely this phone's network blocking the port.
                if (runningSince == 0) {
                    runningSince = System.currentTimeMillis();
                }
                unreachable = System.currentTimeMillis() - runningSince > 60_000;
            }
            if (unreachable) {
                progress("the server is up but this phone cannot reach " + address
                    + " -- is this network blocking odd ports? Try mobile data or another wifi");
            } else {
                String detail = instance.optString("status_msg", "").trim();
                progress(String.format(Locale.US, "server %s%s (%ds)", "null".equals(status) ? "starting" : status,
                    detail.isEmpty() || "null".equals(detail) ? "" : ": " + detail.substring(0, Math.min(80, detail.length())),
                    waited));
            }
            // A pull takes two or three minutes; past ten something is wrong with the machine.
            if (waited > 10 * 60) {
                vast.destroy(id);
                prefs.edit().remove(PREF_INSTANCE).apply();
                throw new IOException("the machine took too long to start, trying another");
            }
            Thread.sleep(4000);
        }
        return null;
    }

    private static String address(JSONObject instance) {
        String ip = instance.optString("public_ipaddr", "").trim();
        JSONObject ports = instance.optJSONObject("ports");
        JSONArray mapped = ports != null ? ports.optJSONArray(Vast.PORT + "/tcp") : null;
        if (ip.isEmpty() || mapped == null || mapped.length() == 0) {
            return null;
        }
        String port = mapped.optJSONObject(0).optString("HostPort", "");
        return port.isEmpty() ? null : ip + ":" + port;
    }

    private boolean healthy(String address) {
        Request request = new Request.Builder().url("http://" + address + "/health").build();
        try (Response response = http.newBuilder().callTimeout(5, TimeUnit.SECONDS).build()
                .newCall(request).execute()) {
            return response.isSuccessful();
        } catch (IOException e) {
            return false;
        }
    }

    private void connect(String address, String token) {
        progress("connecting to the server");
        Request request = new Request.Builder().url("ws://" + address + "/ws?token=" + token).build();
        socket = http.newWebSocket(request, new WebSocketListener() {
            @Override
            public void onOpen(WebSocket ws, Response response) {
                connected = true;
                bringingUp = false;
                progress("connected, starting the engine");
                if (wantSize > 0) {
                    send(message("start").put("size", wantSize));
                }
                send(message("ping"));
            }

            @Override
            public void onMessage(WebSocket ws, String text) {
                try {
                    JSONObject message = new JSONObject(text);
                    switch (message.optString("t")) {
                        case "line":
                            listener.onEngineLine(message.getString("l"));
                            break;
                        case "log":
                            listener.onEngineLog(message.getString("l"));
                            break;
                        case "pong":
                            int left = message.optInt("left");
                            listener.onInfo(info + String.format(Locale.US, " · idle kill in %d:%02d",
                                left / 60, left % 60));
                            break;
                        default:
                    }
                } catch (JSONException e) {
                    // Not ours.
                }
            }

            @Override
            public void onClosed(WebSocket ws, int code, String reason) {
                lost(ws, reason);
            }

            @Override
            public void onFailure(WebSocket ws, Throwable t, Response response) {
                lost(ws, String.valueOf(t.getMessage()));
            }
        });
    }

    private void lost(WebSocket ws, String why) {
        if (ws != socket) {
            return;
        }
        socket = null;
        connected = false;
        bringingUp = false;
        if (!quitting) {
            listener.onInfo(info + " · connection lost (" + why + "), reconnecting");
            worker.schedule(this::ensureUp, 2, TimeUnit.SECONDS);
        }
    }

    // ---- small things -------------------------------------------------------------------

    private void progress(String text) {
        listener.onProgress(text);
        listener.onInfo(info.isEmpty() ? text : info + " · " + text);
    }

    private void setInfo(String text) {
        info = text;
        listener.onInfo(text);
    }

    private void send(Message message) {
        WebSocket ws = socket;
        if (ws != null) {
            ws.send(message.json.toString());
        }
    }

    private static Message message(String type) {
        return new Message(type);
    }

    /** A JSONObject whose put does not throw, for the handful of messages built here. */
    private static final class Message {
        final JSONObject json = new JSONObject();

        Message(String type) {
            put("t", type);
        }

        Message put(String key, Object value) {
            try {
                json.put(key, value);
            } catch (JSONException e) {
                throw new IllegalArgumentException(e);
            }
            return this;
        }
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
