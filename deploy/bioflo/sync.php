<?php
/**
 * CerebroSpheroid Multi-Model Synchronization Hub
 * Hosted on bioflo.in
 *
 * Strict Bounded Storage Policy:
 * Supports 3 model tiers: `v5_small` (101M), `v5_compact` (244M), `v5_base` (617M).
 * Each model tier maintains strictly bounded slots: `{model}_latest.pt` and `{model}_best.pt`.
 * Employs chunked streaming to bypass Cloudflare 100 MB payload limits and PHP upload caps.
 */

// Disable execution time limit for large stream transfers
set_time_limit(600);
ini_set('memory_limit', '256M');

$SECRET_TOKEN = 'CerebroSpheroid_Sync_Secure_2026';

// 1. Authorization Check
$auth_header = '';
if (isset($_SERVER['HTTP_AUTHORIZATION'])) {
    $auth_header = trim($_SERVER['HTTP_AUTHORIZATION']);
} elseif (isset($_SERVER['REDIRECT_HTTP_AUTHORIZATION'])) {
    $auth_header = trim($_SERVER['REDIRECT_HTTP_AUTHORIZATION']);
} elseif (function_exists('apache_request_headers')) {
    $requestHeaders = apache_request_headers();
    $requestHeaders = array_combine(array_map('ucwords', array_keys($requestHeaders)), array_values($requestHeaders));
    if (isset($requestHeaders['Authorization'])) {
        $auth_header = trim($requestHeaders['Authorization']);
    }
}

$token_provided = '';
if (preg_match('/Bearer\s+(.*)$/i', $auth_header, $matches)) {
    $token_provided = $matches[1];
} elseif (isset($_REQUEST['token'])) {
    $token_provided = $_REQUEST['token'];
}

$action = isset($_REQUEST['action']) ? $_REQUEST['action'] : 'status';
$public_actions = ['dashboard', 'tracker', 'view', 'get_tokens', 'tokens'];

if (!in_array($action, $public_actions) && $token_provided !== $SECRET_TOKEN) {
    http_response_code(401);
    header('Content-Type: application/json');
    echo json_encode(['error' => 'Unauthorized: Invalid or missing sync token']);
    exit;
}

// 2. Storage Setup
$storage_dir = __DIR__ . '/data';
if (!is_dir($storage_dir)) {
    mkdir($storage_dir, 0755, true);
    // Protect data directory from direct unauthenticated browser downloads
    file_put_contents($storage_dir . '/.htaccess', "Order Deny,Allow\nDeny from all\n");
}

function normalize_model_name($model) {
    if (empty($model)) return '';
    $clean = preg_replace('/[^a-zA-Z0-9_\-]/', '', strtolower($model));
    if ($clean === 'small' || $clean === 'v5_small') return 'v5_small';
    if ($clean === 'compact' || $clean === 'v5_compact') return 'v5_compact';
    if ($clean === 'base' || $clean === 'v5_base') return 'v5_base';
    return $clean;
}

function get_model_filename($model, $type) {
    if ($model === 'gemma2cerebro' || $type === 'gemma2cerebro') {
        return 'gemma2cerebro.pt';
    }
    $valid_types = ['latest', 'best'];
    $clean_type = in_array($type, $valid_types) ? $type : 'latest';
    $norm_model = normalize_model_name($model);
    if (!empty($norm_model)) {
        return $norm_model . '_' . $clean_type . '.pt';
    }
    return $clean_type . '.pt';
}

$action = isset($_REQUEST['action']) ? $_REQUEST['action'] : 'status';

// --------------------------------------------------------------------------------------------
// Action: STATUS
// --------------------------------------------------------------------------------------------
if ($action === 'status') {
    header('Content-Type: application/json');
    $manifest_file = $storage_dir . '/manifest.json';
    $manifest = (file_exists($manifest_file) && filesize($manifest_file) > 2) ? json_decode(file_get_contents($manifest_file), true) : [];
    if (!is_array($manifest)) {
        $manifest = [];
    }

    $models = ['v5_small', 'v5_compact', 'v5_base'];
    $model_status = [];
    foreach ($models as $m) {
        $l_file = $storage_dir . '/' . $m . '_latest.pt';
        $b_file = $storage_dir . '/' . $m . '_best.pt';
        $has_l = file_exists($l_file);
        $has_b = file_exists($b_file);
        $hist_file = $storage_dir . '/loss_history_' . $m . '.json';
        $has_hist = file_exists($hist_file) && filesize($hist_file) > 2;
        $hist_count = 0;
        if ($has_hist) {
            $h_data = json_decode(file_get_contents($hist_file), true);
            if (is_array($h_data)) $hist_count = count($h_data);
        }
        $model_status[$m] = [
            'has_latest' => $has_l,
            'has_best' => $has_b,
            'latest_size' => $has_l ? filesize($l_file) : 0,
            'best_size' => $has_b ? filesize($b_file) : 0,
            'latest_manifest' => isset($manifest[$m . '_latest']) ? $manifest[$m . '_latest'] : null,
            'best_manifest' => isset($manifest[$m . '_best']) ? $manifest[$m . '_best'] : null,
            'has_loss_history' => $has_hist,
            'loss_history_count' => $hist_count,
        ];
    }

    $req_model = isset($_REQUEST['model']) ? normalize_model_name($_REQUEST['model']) : '';
    if (!empty($req_model) && isset($model_status[$req_model])) {
        $has_latest = $model_status[$req_model]['has_latest'];
        $has_best = $model_status[$req_model]['has_best'];
        $latest_size = $model_status[$req_model]['latest_size'];
        $best_size = $model_status[$req_model]['best_size'];
    } else {
        $legacy_l = file_exists($storage_dir . '/latest.pt');
        $legacy_b = file_exists($storage_dir . '/best.pt');
        $has_latest = $legacy_l || $model_status['v5_small']['has_latest'] || $model_status['v5_compact']['has_latest'];
        $has_best = $legacy_b || $model_status['v5_small']['has_best'] || $model_status['v5_compact']['has_best'];
        $latest_size = $legacy_l ? filesize($storage_dir . '/latest.pt') : 0;
        $best_size = $legacy_b ? filesize($storage_dir . '/best.pt') : 0;
    }

    echo json_encode([
        'status' => 'online',
        'model' => $req_model,
        'has_latest' => $has_latest,
        'has_best' => $has_best,
        'latest_size' => $latest_size,
        'best_size' => $best_size,
        'models' => $model_status,
        'manifest' => $manifest,
        'server_time' => date('c'),
    ]);
    exit;
}

// --------------------------------------------------------------------------------------------
// Action: GET_CONFIG (Retrieve cached hardware-optimized hyperparameters)
// --------------------------------------------------------------------------------------------
if ($action === 'get_config' || $action === 'config') {
    header('Content-Type: application/json');
    $model = isset($_REQUEST['model']) ? normalize_model_name($_REQUEST['model']) : 'v5_small';
    $device = isset($_REQUEST['device']) ? preg_replace('/[^a-zA-Z0-9_\-\.]/', '', $_REQUEST['device']) : 'default';

    $configs_file = $storage_dir . '/hardware_configs.json';
    $configs = (file_exists($configs_file) && filesize($configs_file) > 2) ? json_decode(file_get_contents($configs_file), true) : [];
    if (!is_array($configs)) {
        $configs = [];
    }

    if (isset($configs[$model][$device])) {
        echo json_encode([
            'status' => 'found',
            'model' => $model,
            'device' => $device,
            'config' => $configs[$model][$device],
            'server_time' => date('c'),
        ]);
    } elseif (isset($configs[$model]['default'])) {
        echo json_encode([
            'status' => 'found',
            'model' => $model,
            'device' => 'default',
            'config' => $configs[$model]['default'],
            'server_time' => date('c'),
        ]);
    } else {
        echo json_encode([
            'status' => 'not_found',
            'model' => $model,
            'device' => $device,
            'available_devices' => isset($configs[$model]) ? array_keys($configs[$model]) : [],
            'server_time' => date('c'),
        ]);
    }
    exit;
}

// --------------------------------------------------------------------------------------------
// Action: SAVE_CONFIG (Store tested hardware-optimized hyperparameters)
// --------------------------------------------------------------------------------------------
if ($action === 'save_config') {
    header('Content-Type: application/json');
    
    // Support JSON body or standard form POST
    $input_json = file_get_contents('php://input');
    $payload = (!empty($input_json)) ? json_decode($input_json, true) : null;
    if (!is_array($payload)) {
        $payload = $_POST;
    }

    $model = isset($payload['model']) ? normalize_model_name($payload['model']) : 'v5_small';
    $device = isset($payload['device']) ? preg_replace('/[^a-zA-Z0-9_\-\.]/', '', $payload['device']) : 'default';

    $batch_size = isset($payload['batch_size']) ? intval($payload['batch_size']) : 16;
    $remat = isset($payload['remat']) ? filter_var($payload['remat'], FILTER_VALIDATE_BOOLEAN) : true;
    $seq_len = isset($payload['seq_len']) ? intval($payload['seq_len']) : 256;
    $dtype = isset($payload['dtype']) ? preg_replace('/[^a-zA-Z0-9_\-]/', '', $payload['dtype']) : 'auto';
    $tok_per_sec = isset($payload['tok_per_sec']) ? floatval($payload['tok_per_sec']) : 0.0;
    $step_ms = isset($payload['step_ms']) ? floatval($payload['step_ms']) : 0.0;
    $vram_gb = isset($payload['vram_gb']) ? floatval($payload['vram_gb']) : 0.0;
    $vram_utilization_pct = isset($payload['vram_utilization_pct']) ? floatval($payload['vram_utilization_pct']) : 0.0;
    $source = isset($payload['source']) ? preg_replace('/[^a-zA-Z0-9_\-]/', '', $payload['source']) : 'colab';

    $entry = [
        'model' => $model,
        'device' => $device,
        'batch_size' => $batch_size,
        'remat' => $remat,
        'seq_len' => $seq_len,
        'dtype' => $dtype,
        'tok_per_sec' => $tok_per_sec,
        'step_ms' => $step_ms,
        'vram_gb' => $vram_gb,
        'vram_utilization_pct' => $vram_utilization_pct,
        'source' => $source,
        'updated_at' => date('c'),
    ];

    $configs_file = $storage_dir . '/hardware_configs.json';
    $configs = (file_exists($configs_file) && filesize($configs_file) > 2) ? json_decode(file_get_contents($configs_file), true) : [];
    if (!is_array($configs)) {
        $configs = [];
    }
    if (!isset($configs[$model])) {
        $configs[$model] = [];
    }
    $configs[$model][$device] = $entry;

    file_put_contents($configs_file, json_encode($configs, JSON_PRETTY_PRINT));

    echo json_encode([
        'success' => true,
        'status' => 'saved',
        'model' => $model,
        'device' => $device,
        'config' => $entry,
    ]);
    exit;
}

// --------------------------------------------------------------------------------------------
// Action: SAVE_TOKENS (Persist real-time token ingestion counters and per-dataset attribution)
// --------------------------------------------------------------------------------------------
if ($action === 'save_tokens' || $action === 'record_tokens') {
    header('Content-Type: application/json');

    $input_data = null;
    $raw_input = file_get_contents('php://input');
    if (!empty($raw_input)) {
        $json = json_decode($raw_input, true);
        if (is_array($json)) {
            $input_data = $json;
        }
    }
    if (!$input_data) {
        $input_data = $_POST;
    }

    $model = isset($input_data['model']) ? normalize_model_name($input_data['model']) : 'v5_small';
    if (empty($model)) $model = 'v5_small';

    $cumulative_tokens = isset($input_data['cumulative_tokens']) ? floatval($input_data['cumulative_tokens']) : 0;
    $target_tokens = isset($input_data['target_tokens']) ? floatval($input_data['target_tokens']) : 200000000000;
    $tokens_by_dataset = isset($input_data['tokens_by_dataset']) && is_array($input_data['tokens_by_dataset']) ? $input_data['tokens_by_dataset'] : [];
    $effective_batch_size = isset($input_data['effective_batch_size']) ? intval($input_data['effective_batch_size']) : 256;
    $step = isset($input_data['step']) ? intval($input_data['step']) : 0;
    $loss = isset($input_data['loss']) ? floatval($input_data['loss']) : 0.0;
    $tok_per_sec = isset($input_data['tok_per_sec']) ? floatval($input_data['tok_per_sec']) : 0.0;
    $source = isset($input_data['source']) ? preg_replace('/[^a-zA-Z0-9_\-]/', '', $input_data['source']) : 'unknown';

    $tracker_file = $storage_dir . '/token_tracker_' . $model . '.json';

    // Monotonic Token Ingestion Safeguard: Prevent restarted sessions from resetting global token accounting
    $force_reset = isset($input_data['force_reset']) && filter_var($input_data['force_reset'], FILTER_VALIDATE_BOOLEAN);
    if (!$force_reset && file_exists($tracker_file)) {
        $old_tracker = json_decode(file_get_contents($tracker_file), true);
        if (is_array($old_tracker)) {
            $old_cum = isset($old_tracker['cumulative_tokens']) ? floatval($old_tracker['cumulative_tokens']) : 0;
            if ($old_cum > $cumulative_tokens && $step < 10000) {
                // Worker restarted from Step 0/scratch: accumulate new tokens on top of preserved historical baseline
                $cumulative_tokens = $old_cum + $cumulative_tokens;
            }
        }
    }

    $tracker_data = [
        'model' => $model,
        'cumulative_tokens' => $cumulative_tokens,
        'target_tokens' => $target_tokens,
        'pct_complete' => ($target_tokens > 0) ? round(($cumulative_tokens / $target_tokens) * 100, 4) : 0,
        'tokens_by_dataset' => $tokens_by_dataset,
        'effective_batch_size' => $effective_batch_size,
        'step' => $step,
        'loss' => $loss,
        'tok_per_sec' => $tok_per_sec,
        'source' => $source,
        'updated_at' => date('c'),
        'timestamp' => time()
    ];

    file_put_contents($tracker_file, json_encode($tracker_data, JSON_PRETTY_PRINT));

    // Update manifest
    $manifest_file = $storage_dir . '/manifest.json';
    $manifest = (file_exists($manifest_file) && filesize($manifest_file) > 2) ? json_decode(file_get_contents($manifest_file), true) : [];
    if (!is_array($manifest)) $manifest = [];
    $manifest['token_tracker_' . $model] = [
        'model' => $model,
        'cumulative_tokens' => $cumulative_tokens,
        'target_tokens' => $target_tokens,
        'step' => $step,
        'effective_batch_size' => $effective_batch_size,
        'updated_at' => date('c'),
    ];
    file_put_contents($manifest_file, json_encode($manifest, JSON_PRETTY_PRINT));

    echo json_encode([
        'success' => true,
        'tracker' => $tracker_data
    ]);
    exit;
}

// --------------------------------------------------------------------------------------------
// Action: GET_TOKENS (Retrieve current token ingestion stats and breakdown)
// --------------------------------------------------------------------------------------------
if ($action === 'get_tokens' || $action === 'tokens') {
    header('Content-Type: application/json');
    $model = isset($_REQUEST['model']) ? normalize_model_name($_REQUEST['model']) : 'v5_small';
    if (empty($model)) $model = 'v5_small';

    $tracker_file = $storage_dir . '/token_tracker_' . $model . '.json';
    if (file_exists($tracker_file) && filesize($tracker_file) > 2) {
        $data = json_decode(file_get_contents($tracker_file), true);
        if (is_array($data)) {
            echo json_encode(['success' => true, 'data' => $data]);
            exit;
        }
    }

    echo json_encode([
        'success' => true,
        'data' => [
            'model' => $model,
            'cumulative_tokens' => 0,
            'target_tokens' => 200000000000,
            'pct_complete' => 0.0,
            'tokens_by_dataset' => [
                'fineweb_edu' => 0,
                'cosmopedia' => 0,
                'finemath' => 0,
                'python_code' => 0,
                'smoltalk' => 0
            ],
            'effective_batch_size' => 256,
            'step' => 0,
            'loss' => 0.0,
            'tok_per_sec' => 0.0,
            'source' => 'idle',
            'updated_at' => null,
            'timestamp' => 0
        ]
    ]);
    exit;
}

// --------------------------------------------------------------------------------------------
// Action: DASHBOARD / TRACKER (Live interactive UI for 200B training mission control)
// --------------------------------------------------------------------------------------------
if ($action === 'dashboard' || $action === 'tracker' || $action === 'view') {
    $model = isset($_REQUEST['model']) ? normalize_model_name($_REQUEST['model']) : 'v5_small';
    if (empty($model)) $model = 'v5_small';
    header('Content-Type: text/html; charset=utf-8');
?>
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>CerebroSpheroid V5 - 200B Token Pre-Training Monitor</title>
    <style>
        :root {
            --bg-base: #080c14;
            --bg-card: rgba(17, 24, 39, 0.85);
            --bg-card-hover: rgba(30, 41, 59, 0.95);
            --border-card: rgba(56, 189, 248, 0.15);
            --border-glow: rgba(56, 189, 248, 0.4);
            --text-main: #f8fafc;
            --text-muted: #94a3b8;
            --accent-cyan: #38bdf8;
            --accent-purple: #a855f7;
            --accent-emerald: #10b981;
            --accent-amber: #f59e0b;
            --accent-rose: #f43f5e;
        }
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
            background: radial-gradient(circle at 50% 0%, #0f172a 0%, #080c14 100%);
            color: var(--text-main);
            min-height: 100vh;
            padding: 24px 16px;
        }
        .container { max-width: 1200px; margin: 0 auto; }
        header {
            display: flex;
            flex-wrap: wrap;
            justify-content: space-between;
            align-items: center;
            gap: 16px;
            margin-bottom: 24px;
            padding-bottom: 16px;
            border-bottom: 1px solid rgba(255, 255, 255, 0.1);
        }
        .logo-area h1 {
            font-size: 1.75rem;
            font-weight: 800;
            background: linear-gradient(135deg, #38bdf8, #818cf8, #c084fc);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            display: flex;
            align-items: center;
            gap: 10px;
        }
        .logo-area p {
            color: var(--text-muted);
            font-size: 0.9rem;
            margin-top: 4px;
        }
        .status-pill {
            display: inline-flex;
            align-items: center;
            gap: 8px;
            background: rgba(16, 185, 129, 0.12);
            border: 1px solid rgba(16, 185, 129, 0.3);
            color: #34d399;
            padding: 6px 14px;
            border-radius: 9999px;
            font-size: 0.85rem;
            font-weight: 600;
        }
        .status-dot {
            width: 8px;
            height: 8px;
            border-radius: 50%;
            background: #10b981;
            box-shadow: 0 0 10px #10b981;
            animation: pulse-beacon 2s infinite ease-in-out;
        }
        @keyframes pulse-beacon {
            0%, 100% { opacity: 1; transform: scale(1); }
            50% { opacity: 0.4; transform: scale(1.3); }
        }
        .model-tabs {
            display: flex;
            gap: 8px;
            margin-bottom: 24px;
            background: rgba(15, 23, 42, 0.7);
            padding: 6px;
            border-radius: 12px;
            width: fit-content;
            border: 1px solid rgba(255, 255, 255, 0.08);
        }
        .model-tab {
            padding: 8px 18px;
            border-radius: 8px;
            text-decoration: none;
            color: var(--text-muted);
            font-size: 0.9rem;
            font-weight: 600;
            transition: all 0.2s;
        }
        .model-tab:hover { color: #fff; background: rgba(255, 255, 255, 0.05); }
        .model-tab.active {
            background: linear-gradient(135deg, rgba(56, 189, 248, 0.2), rgba(168, 85, 247, 0.2));
            color: #38bdf8;
            border: 1px solid var(--border-glow);
        }
        /* Hero Progress Banner */
        .hero-banner {
            background: var(--bg-card);
            backdrop-filter: blur(12px);
            border: 1px solid var(--border-card);
            border-radius: 18px;
            padding: 28px 24px;
            margin-bottom: 24px;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.4);
            position: relative;
            overflow: hidden;
        }
        .hero-banner::after {
            content: '';
            position: absolute;
            top: 0; right: 0; bottom: 0; left: 0;
            background: radial-gradient(circle at 90% 10%, rgba(56, 189, 248, 0.06), transparent 50%);
            pointer-events: none;
        }
        .hero-top {
            display: flex;
            flex-wrap: wrap;
            justify-content: space-between;
            align-items: baseline;
            gap: 12px;
            margin-bottom: 12px;
        }
        .hero-label {
            font-size: 0.95rem;
            font-weight: 700;
            letter-spacing: 0.05em;
            text-transform: uppercase;
            color: var(--accent-cyan);
        }
        .hero-stat {
            font-size: 2.5rem;
            font-weight: 900;
            letter-spacing: -0.02em;
        }
        .hero-stat span {
            font-size: 1.25rem;
            color: var(--text-muted);
            font-weight: 500;
        }
        .progress-bar-container {
            width: 100%;
            height: 22px;
            background: rgba(15, 23, 42, 0.9);
            border-radius: 9999px;
            overflow: hidden;
            position: relative;
            border: 1px solid rgba(255, 255, 255, 0.1);
            margin: 16px 0;
        }
        .progress-bar-fill {
            height: 100%;
            width: 0%;
            background: linear-gradient(90deg, #38bdf8 0%, #818cf8 40%, #c084fc 80%, #34d399 100%);
            border-radius: 9999px;
            box-shadow: 0 0 16px rgba(56, 189, 248, 0.6);
            transition: width 1s cubic-bezier(0.4, 0, 0.2, 1);
        }
        .hero-substats {
            display: flex;
            flex-wrap: wrap;
            gap: 24px;
            margin-top: 14px;
            font-size: 0.9rem;
            color: var(--text-muted);
        }
        .hero-substats strong { color: var(--text-main); }
        /* Grid of Metrics */
        .metrics-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
            gap: 16px;
            margin-bottom: 28px;
        }
        .metric-card {
            background: var(--bg-card);
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 14px;
            padding: 18px;
            transition: all 0.2s;
        }
        .metric-card:hover {
            border-color: var(--border-glow);
            transform: translateY(-2px);
        }
        .metric-title {
            font-size: 0.8rem;
            text-transform: uppercase;
            font-weight: 700;
            color: var(--text-muted);
            margin-bottom: 8px;
        }
        .metric-value {
            font-size: 1.5rem;
            font-weight: 800;
            color: var(--text-main);
        }
        .metric-desc {
            font-size: 0.78rem;
            color: var(--text-muted);
            margin-top: 4px;
        }
        /* Dataset Breakdown Cards */
        .section-title {
            font-size: 1.25rem;
            font-weight: 800;
            margin-bottom: 16px;
            display: flex;
            align-items: center;
            gap: 8px;
        }
        .dataset-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
            gap: 16px;
            margin-bottom: 32px;
        }
        .dataset-card {
            background: var(--bg-card);
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 14px;
            padding: 20px;
            position: relative;
            overflow: hidden;
        }
        .dataset-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 10px;
        }
        .dataset-name {
            font-weight: 700;
            font-size: 1.05rem;
        }
        .dataset-badge {
            font-size: 0.75rem;
            padding: 3px 8px;
            border-radius: 6px;
            font-weight: 600;
        }
        .dataset-tokens {
            font-size: 1.7rem;
            font-weight: 800;
            margin: 6px 0;
        }
        .dataset-pct {
            font-size: 0.85rem;
            color: var(--text-muted);
            margin-bottom: 12px;
        }
        .mini-progress {
            width: 100%;
            height: 8px;
            background: rgba(15, 23, 42, 0.8);
            border-radius: 9999px;
            overflow: hidden;
            margin-top: 8px;
        }
        .mini-progress-fill {
            height: 100%;
            border-radius: 9999px;
            transition: width 1s ease-in-out;
        }
        /* Curriculum Roadmap */
        .curriculum-banner {
            background: var(--bg-card);
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 14px;
            padding: 22px;
            margin-bottom: 28px;
        }
        .curriculum-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
            gap: 12px;
            margin-top: 14px;
        }
        .phase-box {
            background: rgba(15, 23, 42, 0.6);
            border: 1px solid rgba(255, 255, 255, 0.05);
            border-radius: 10px;
            padding: 14px;
        }
        .phase-box.current {
            border-color: var(--accent-cyan);
            background: rgba(56, 189, 248, 0.05);
        }
        .phase-name { font-weight: 700; font-size: 0.9rem; color: var(--accent-cyan); }
        .phase-tokens { font-size: 0.8rem; color: var(--text-muted); margin: 4px 0; }
        .phase-mix { font-size: 0.75rem; color: #cbd5e1; line-height: 1.4; }
        footer {
            text-align: center;
            font-size: 0.85rem;
            color: var(--text-muted);
            margin-top: 36px;
            padding-top: 16px;
            border-top: 1px solid rgba(255, 255, 255, 0.08);
        }
        .btn-refresh {
            background: rgba(56, 189, 248, 0.15);
            border: 1px solid rgba(56, 189, 248, 0.4);
            color: var(--accent-cyan);
            padding: 6px 14px;
            border-radius: 8px;
            cursor: pointer;
            font-size: 0.85rem;
            font-weight: 600;
            transition: all 0.2s;
        }
        .btn-refresh:hover { background: rgba(56, 189, 248, 0.25); }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <div class="logo-area">
                <h1>🧠 CerebroSpheroid V5</h1>
                <p>200 Billion Token Pre-Training Monitor & Per-Dataset Attribution</p>
            </div>
            <div style="display: flex; align-items: center; gap: 12px;">
                <button class="btn-refresh" onclick="fetchLiveStats()">🔄 Refresh</button>
                <div class="status-pill">
                    <div class="status-dot"></div>
                    <span id="liveStatusText">TPU v5e-1 STREAMING</span>
                </div>
            </div>
        </header>

        <div class="model-tabs">
            <a href="?action=dashboard&model=v5_small" class="model-tab <?php echo ($model === 'v5_small' ? 'active' : ''); ?>">v5_small (101.96M)</a>
            <a href="?action=dashboard&model=v5_compact" class="model-tab <?php echo ($model === 'v5_compact' ? 'active' : ''); ?>">v5_compact (244M)</a>
            <a href="?action=dashboard&model=v5_base" class="model-tab <?php echo ($model === 'v5_base' ? 'active' : ''); ?>">v5_base (617M)</a>
        </div>

        <!-- 200B Hero Banner -->
        <div class="hero-banner">
            <div class="hero-top">
                <div class="hero-label">200.00 Billion Token Campaign Progress</div>
                <div class="hero-stat">
                    <span id="heroTokens">0.000</span> / 200.00 <span style="font-size: 1.5rem; color: #fff;">B</span>
                    <span id="heroPct" style="margin-left: 12px; color: var(--accent-cyan); font-weight: 700;">(0.00%)</span>
                </div>
            </div>
            <div class="progress-bar-container">
                <div id="heroProgressBar" class="progress-bar-fill"></div>
            </div>
            <div class="hero-substats">
                <div>Exact Ingested: <strong id="exactTokens">0</strong> tokens</div>
                <div>Remaining: <strong id="remainingTokens">200,000,000,000</strong> tokens</div>
                <div>ETA: <strong id="etaHours">-- hours</strong></div>
                <div>Last Sync: <strong id="lastSyncTime">Just now</strong></div>
            </div>
        </div>

        <!-- Metrics Grid -->
        <div class="metrics-grid">
            <div class="metric-card">
                <div class="metric-title">Effective Batch Size</div>
                <div class="metric-value" id="valBatchSize">256</div>
                <div class="metric-desc">TPU v5e-1 Base (131,072 tok/step)</div>
            </div>
            <div class="metric-card">
                <div class="metric-title">Optimizer Step</div>
                <div class="metric-value" id="valStep">0</div>
                <div class="metric-desc" id="valTotalSteps">Target: 1,525,879 steps</div>
            </div>
            <div class="metric-card">
                <div class="metric-title">Sustained Speed</div>
                <div class="metric-value" id="valTokSpeed">0 tok/s</div>
                <div class="metric-desc" id="valStepSpeed">-- steps/s</div>
            </div>
            <div class="metric-card">
                <div class="metric-title">Training Loss</div>
                <div class="metric-value" id="valLoss">--</div>
                <div class="metric-desc">Smooth Manifold Descent</div>
            </div>
            <div class="metric-card">
                <div class="metric-title">Campaign Regime</div>
                <div class="metric-value" style="color: var(--accent-emerald);">1,961</div>
                <div class="metric-desc">Tokens/Param (Deep Saturated)</div>
            </div>
            <div class="metric-card">
                <div class="metric-title">Execution Target</div>
                <div class="metric-value" id="valSource">TPU v5e-1</div>
                <div class="metric-desc">Google Colab Systolic Engine</div>
            </div>
        </div>

        <!-- Dataset Breakdown -->
        <div class="section-title">
            <span>📊</span> Ingestion Breakdown Per Dataset Stream
        </div>
        <div class="dataset-grid">
            <!-- 1. FineWeb-Edu -->
            <div class="dataset-card">
                <div class="dataset-header">
                    <div class="dataset-name">🔬 FineWeb-Edu (score &ge; 3.0)</div>
                    <span class="dataset-badge" style="background: rgba(56, 189, 248, 0.15); color: #38bdf8;">Science & Knowledge</span>
                </div>
                <div class="dataset-tokens" id="dsFineWebTokens">0.00 B</div>
                <div class="dataset-pct" id="dsFineWebPct">0.0% of total ingested</div>
                <div class="mini-progress">
                    <div id="dsFineWebBar" class="mini-progress-fill" style="background: #38bdf8; width: 0%;"></div>
                </div>
            </div>

            <!-- 2. Cosmopedia -->
            <div class="dataset-card">
                <div class="dataset-header">
                    <div class="dataset-name">📚 Cosmopedia v2</div>
                    <span class="dataset-badge" style="background: rgba(168, 85, 247, 0.15); color: #c084fc;">Synthetic Textbooks</span>
                </div>
                <div class="dataset-tokens" id="dsCosmoTokens">0.00 B</div>
                <div class="dataset-pct" id="dsCosmoPct">0.0% of total ingested</div>
                <div class="mini-progress">
                    <div id="dsCosmoBar" class="mini-progress-fill" style="background: #a855f7; width: 0%;"></div>
                </div>
            </div>

            <!-- 3. FineMath & Proofs -->
            <div class="dataset-card">
                <div class="dataset-header">
                    <div class="dataset-name">📐 FineMath 4+ & Proofs</div>
                    <span class="dataset-badge" style="background: rgba(245, 158, 11, 0.15); color: #fbbf24;">Math & CoT</span>
                </div>
                <div class="dataset-tokens" id="dsMathTokens">0.00 B</div>
                <div class="dataset-pct" id="dsMathPct">0.0% of total ingested</div>
                <div class="mini-progress">
                    <div id="dsMathBar" class="mini-progress-fill" style="background: #f59e0b; width: 0%;"></div>
                </div>
            </div>

            <!-- 4. Python Code -->
            <div class="dataset-card">
                <div class="dataset-header">
                    <div class="dataset-name">💻 Python Code Large</div>
                    <span class="dataset-badge" style="background: rgba(16, 185, 129, 0.15); color: #34d399;">Software & Scripts</span>
                </div>
                <div class="dataset-tokens" id="dsCodeTokens">0.00 B</div>
                <div class="dataset-pct" id="dsCodePct">0.0% of total ingested</div>
                <div class="mini-progress">
                    <div id="dsCodeBar" class="mini-progress-fill" style="background: #10b981; width: 0%;"></div>
                </div>
            </div>

            <!-- 5. SmolTalk & Logic -->
            <div class="dataset-card">
                <div class="dataset-header">
                    <div class="dataset-name">💬 SmolTalk & Reasoning</div>
                    <span class="dataset-badge" style="background: rgba(244, 63, 94, 0.15); color: #fb7185;">Commonsense & Logic</span>
                </div>
                <div class="dataset-tokens" id="dsSmolTokens">0.00 B</div>
                <div class="dataset-pct" id="dsSmolPct">0.0% of total ingested</div>
                <div class="mini-progress">
                    <div id="dsSmolBar" class="mini-progress-fill" style="background: #f43f5e; width: 0%;"></div>
                </div>
            </div>
        </div>

        <!-- Curriculum Timeline -->
        <div class="curriculum-banner">
            <div class="section-title" style="margin-bottom: 4px;">
                <span>🎯</span> 4-Phase Developmental Curriculum Progress
            </div>
            <div class="curriculum-grid">
                <div class="phase-box" id="phaseBox1">
                    <div class="phase-name">Phase 1: Concept Imprinting</div>
                    <div class="phase-tokens">0 &rarr; 30B tokens (0% &rarr; 15%)</div>
                    <div class="phase-mix">60% FineWeb-Edu (&ge;3.0), 15% Cosmopedia, 10% Everyday, 10% Math, 5% Code</div>
                </div>
                <div class="phase-box" id="phaseBox2">
                    <div class="phase-name">Phase 2: Commonsense & Code</div>
                    <div class="phase-tokens">30B &rarr; 100B tokens (15% &rarr; 50%)</div>
                    <div class="phase-mix">35% FineWeb-Edu, 25% Commonsense, 20% Python Code, 20% Math Proofs</div>
                </div>
                <div class="phase-box" id="phaseBox3">
                    <div class="phase-name">Phase 3: Tri-Pillar Deliberation</div>
                    <div class="phase-tokens">100B &rarr; 170B tokens (50% &rarr; 85%)</div>
                    <div class="phase-mix">33.3% Science, 33.3% Commonsense & Logic, 33.3% Math-CoT & Indented Code</div>
                </div>
                <div class="phase-box" id="phaseBox4">
                    <div class="phase-name">Phase 4: Basin Consolidation</div>
                    <div class="phase-tokens">170B &rarr; 200B tokens (85% &rarr; 100%)</div>
                    <div class="phase-mix">30% High-Grade Science, 30% Verified Proofs, 20% Python Code, 20% Magpie Logic</div>
                </div>
            </div>
        </div>

        <footer>
            CerebroSpheroid V5 &copy; 2026 &bull; Live Telemetry &bull; Hosted on bioflo.in
        </footer>
    </div>

    <script>
        const CURRENT_MODEL = '<?php echo $model; ?>';

        function formatTokens(num) {
            if (!num || isNaN(num)) return '0.00 B';
            return (num / 1e9).toFixed(3) + ' B';
        }

        function formatNumber(num) {
            if (!num || isNaN(num)) return '0';
            return Number(num).toLocaleString();
        }

        async function fetchLiveStats() {
            try {
                const res = await fetch(`?action=get_tokens&model=${CURRENT_MODEL}&t=${Date.now()}`);
                const json = await res.json();
                if (!json.success || !json.data) return;

                const d = json.data;
                const cumTok = d.cumulative_tokens || 0;
                const tgtTok = d.target_tokens || 200000000000;
                const pct = tgtTok > 0 ? (cumTok / tgtTok) * 100 : 0;

                // Hero elements
                document.getElementById('heroTokens').innerText = (cumTok / 1e9).toFixed(3);
                document.getElementById('heroPct').innerText = `(${pct.toFixed(2)}%)`;
                document.getElementById('heroProgressBar').style.width = Math.min(100, Math.max(0, pct)) + '%';
                document.getElementById('exactTokens').innerText = formatNumber(cumTok);
                document.getElementById('remainingTokens').innerText = formatNumber(Math.max(0, tgtTok - cumTok));

                // ETA
                const speed = d.tok_per_sec || 0;
                if (speed > 0 && cumTok < tgtTok) {
                    const secRemaining = (tgtTok - cumTok) / speed;
                    const hoursRemaining = (secRemaining / 3600).toFixed(1);
                    document.getElementById('etaHours').innerText = `${hoursRemaining} hours (~${(hoursRemaining / 24).toFixed(1)} days)`;
                } else if (cumTok >= tgtTok) {
                    document.getElementById('etaHours').innerText = 'Completed 🏆';
                } else {
                    document.getElementById('etaHours').innerText = '-- hours';
                }

                if (d.updated_at) {
                    const dt = new Date(d.updated_at);
                    document.getElementById('lastSyncTime').innerText = dt.toLocaleTimeString();
                }

                // Metric Cards
                const effBs = d.effective_batch_size || 256;
                document.getElementById('valBatchSize').innerText = effBs;
                document.getElementById('valStep').innerText = formatNumber(d.step || 0);

                const tokPerStep = effBs * 512;
                const totalSteps = Math.ceil(tgtTok / Math.max(1, tokPerStep));
                document.getElementById('valTotalSteps').innerText = `Target: ${formatNumber(totalSteps)} steps`;

                document.getElementById('valTokSpeed').innerText = speed > 0 ? `${formatNumber(Math.round(speed))} tok/s` : '-- tok/s';
                const sps = speed > 0 && tokPerStep > 0 ? (speed / tokPerStep).toFixed(2) : '--';
                document.getElementById('valStepSpeed').innerText = `${sps} steps/s`;

                document.getElementById('valLoss').innerText = d.loss > 0 ? Number(d.loss).toFixed(4) : '--';
                document.getElementById('valSource').innerText = d.source ? d.source.toUpperCase() : 'TPU v5e-1';

                // Dataset Breakdown
                const byDs = d.tokens_by_dataset || {};
                const totalDs = Object.values(byDs).reduce((a, b) => a + Number(b), 0) || cumTok || 1;

                function updateDatasetCard(tokens, tokensElemId, pctElemId, barElemId) {
                    const t = Number(tokens || 0);
                    const p = ((t / totalDs) * 100).toFixed(1);
                    document.getElementById(tokensElemId).innerText = formatTokens(t);
                    document.getElementById(pctElemId).innerText = `${p}% of ingested`;
                    document.getElementById(barElemId).style.width = Math.min(100, Math.max(0, p)) + '%';
                }

                updateDatasetCard(byDs['fineweb_edu'], 'dsFineWebTokens', 'dsFineWebPct', 'dsFineWebBar');
                updateDatasetCard(byDs['cosmopedia'], 'dsCosmoTokens', 'dsCosmoPct', 'dsCosmoBar');
                updateDatasetCard(byDs['finemath'], 'dsMathTokens', 'dsMathPct', 'dsMathBar');
                updateDatasetCard(byDs['python_code'], 'dsCodeTokens', 'dsCodePct', 'dsCodeBar');
                updateDatasetCard(byDs['smoltalk'], 'dsSmolTokens', 'dsSmolPct', 'dsSmolBar');

                // Active Phase Highlight
                for (let i = 1; i <= 4; i++) {
                    const b = document.getElementById(`phaseBox${i}`);
                    if (b) b.classList.remove('current');
                }
                if (pct < 15) document.getElementById('phaseBox1').classList.add('current');
                else if (pct < 50) document.getElementById('phaseBox2').classList.add('current');
                else if (pct < 85) document.getElementById('phaseBox3').classList.add('current');
                else document.getElementById('phaseBox4').classList.add('current');

            } catch (err) {
                console.error("Failed to fetch live stats:", err);
            }
        }

        fetchLiveStats();
        setInterval(fetchLiveStats, 10000);
    </script>
</body>
</html>
<?php
    exit;
}


// --------------------------------------------------------------------------------------------
// Action: GET_LOSS_HISTORY (Retrieve recorded training loss trajectory)
// --------------------------------------------------------------------------------------------
if ($action === 'get_loss_history' || $action === 'loss_history') {
    header('Content-Type: application/json');
    $model = isset($_REQUEST['model']) ? normalize_model_name($_REQUEST['model']) : 'v5_small';
    $norm_m = !empty($model) ? $model : 'v5_small';
    $history_file = $storage_dir . '/loss_history_' . $norm_m . '.json';

    // Fallback to legacy un-prefixed history if model-specific file doesn't exist
    if (!file_exists($history_file) && file_exists($storage_dir . '/loss_history.json')) {
        $history_file = $storage_dir . '/loss_history.json';
    }

    if (file_exists($history_file) && filesize($history_file) > 2) {
        $data = json_decode(file_get_contents($history_file), true);
        if (is_array($data)) {
            echo json_encode([
                'status' => 'found',
                'model' => $norm_m,
                'count' => count($data),
                'records' => $data,
                'server_time' => date('c'),
            ]);
            exit;
        }
    }

    echo json_encode([
        'status' => 'not_found',
        'model' => $norm_m,
        'count' => 0,
        'records' => [],
        'server_time' => date('c'),
    ]);
    exit;
}

// --------------------------------------------------------------------------------------------
// Action: SAVE_LOSS_HISTORY (Save / append loss trajectory data points to website data)
// --------------------------------------------------------------------------------------------
if ($action === 'save_loss_history' || $action === 'save_loss' || $action === 'record_loss') {
    header('Content-Type: application/json');

    $input_json = file_get_contents('php://input');
    $payload = (!empty($input_json)) ? json_decode($input_json, true) : null;
    if (!is_array($payload)) {
        $payload = $_POST;
    }

    $model = isset($payload['model']) ? normalize_model_name($payload['model']) : (isset($_REQUEST['model']) ? normalize_model_name($_REQUEST['model']) : 'v5_small');
    $norm_m = !empty($model) ? $model : 'v5_small';
    $history_file = $storage_dir . '/loss_history_' . $norm_m . '.json';

    $existing = [];
    if (file_exists($history_file) && filesize($history_file) > 2) {
        $decoded = json_decode(file_get_contents($history_file), true);
        if (is_array($decoded)) {
            $existing = $decoded;
        }
    }

    // Map existing records by step for deduplication
    $by_step = [];
    foreach ($existing as $r) {
        if (isset($r['step'])) {
            $by_step[intval($r['step'])] = $r;
        }
    }

    // Records can be passed as a list under 'records' or as a single record payload
    $incoming_records = [];
    if (isset($payload['records']) && is_array($payload['records'])) {
        $incoming_records = $payload['records'];
    } elseif (isset($payload['step']) && isset($payload['loss'])) {
        $incoming_records = [$payload];
    }

    $added_or_updated = 0;
    foreach ($incoming_records as $rec) {
        if (!isset($rec['step']) || !isset($rec['loss'])) {
            continue;
        }
        $s = intval($rec['step']);
        $clean_rec = [
            'step' => $s,
            'loss' => floatval($rec['loss']),
            'eval_loss' => (isset($rec['eval_loss']) && $rec['eval_loss'] !== null) ? floatval($rec['eval_loss']) : null,
            'ema_loss' => (isset($rec['ema_loss']) && $rec['ema_loss'] !== null) ? floatval($rec['ema_loss']) : null,
            'delta' => (isset($rec['delta']) && $rec['delta'] !== null) ? floatval($rec['delta']) : null,
            'status' => isset($rec['status']) ? preg_replace('/[^a-zA-Z0-9_\-]/', '', strval($rec['status'])) : 'SMOOTH_DESCENT',
            'timestamp' => isset($rec['timestamp']) && !empty($rec['timestamp']) ? strval($rec['timestamp']) : date('c'),
            'source' => isset($rec['source']) ? preg_replace('/[^a-zA-Z0-9_\-]/', '', strval($rec['source'])) : 'cloud',
        ];
        if (isset($rec['md5'])) {
            $clean_rec['md5'] = strval($rec['md5']);
        }
        if (isset($rec['size_bytes'])) {
            $clean_rec['size_bytes'] = intval($rec['size_bytes']);
        }

        // Merge if existing record had eval_loss and new doesn't, preserve eval_loss
        if (isset($by_step[$s])) {
            if ($clean_rec['eval_loss'] === null && isset($by_step[$s]['eval_loss']) && $by_step[$s]['eval_loss'] !== null) {
                $clean_rec['eval_loss'] = $by_step[$s]['eval_loss'];
            }
        }

        $by_step[$s] = $clean_rec;
        $added_or_updated++;
    }

    // Sort by step ascending
    ksort($by_step);
    $merged = array_values($by_step);

    // Limit to latest 10,000 points to keep storage strictly bounded
    if (count($merged) > 10000) {
        $merged = array_slice($merged, -10000);
    }

    file_put_contents($history_file, json_encode($merged, JSON_PRETTY_PRINT));

    echo json_encode([
        'success' => true,
        'status' => 'saved',
        'model' => $norm_m,
        'added_or_updated' => $added_or_updated,
        'total_records' => count($merged),
        'server_time' => date('c'),
    ]);
    exit;
}

// --------------------------------------------------------------------------------------------
// Action: NOTIFY / TRAINING_START
// --------------------------------------------------------------------------------------------
if ($action === 'notify' || $action === 'start' || $action === 'training_start') {
    header('Content-Type: application/json');
    $source = isset($_REQUEST['source']) ? preg_replace('/[^a-zA-Z0-9_\-]/', '', $_REQUEST['source']) : 'colab';
    $model = isset($_REQUEST['model']) ? normalize_model_name($_REQUEST['model']) : 'v5_compact';
    $message = isset($_REQUEST['message']) ? trim($_REQUEST['message']) : "🚀 CerebroSpheroid {$model} training active on {$source}";
    $step = isset($_REQUEST['step']) ? intval($_REQUEST['step']) : 0;
    $hours = isset($_REQUEST['hours']) ? floatval($_REQUEST['hours']) : 8.5;

    $manifest_file = $storage_dir . '/manifest.json';
    $manifest = file_exists($manifest_file) ? json_decode(file_get_contents($manifest_file), true) : [];
    if (!is_array($manifest)) {
        $manifest = [];
    }

    $session_key = !empty($model) ? "active_session_{$model}" : "active_session";
    $manifest[$session_key] = [
        'model' => $model,
        'source' => $source,
        'status' => 'training',
        'message' => $message,
        'step' => $step,
        'max_hours' => $hours,
        'started_at' => date('c'),
        'last_ping' => date('c'),
    ];
    file_put_contents($manifest_file, json_encode($manifest, JSON_PRETTY_PRINT));

    // Send email alert to user so they know training has started
    $to = 'meajitsheoran@gmail.com';
    $subject = "🚀 [BioFlo AI] CerebroSpheroid {$model} Training Started on {$source}";
    $body = "Hi Ajit,\n\n"
          . "CerebroSpheroid V5 ({$model}) training has just started on {$source}.\n\n"
          . "Details:\n"
          . "• Model Tier: {$model}\n"
          . "• Node: {$source}\n"
          . "• Starting Step: " . number_format($step) . "\n"
          . "• Planned Window: {$hours} hours\n"
          . "• Time: " . date('Y-m-d H:i:s T') . "\n"
          . "• Message: {$message}\n\n"
          . "BioFlo AI Training Cluster";

    $headers = "From: BioFlo AI <notifications@bioflo.in>\r\n"
             . "Reply-To: notifications@bioflo.in\r\n"
             . "X-Mailer: PHP/" . phpversion();

    @mail($to, $subject, $body, $headers);

    echo json_encode([
        'success' => true,
        'notified' => true,
        'model' => $model,
        'source' => $source,
        'active_session' => $manifest[$session_key]
    ]);
    exit;
}

// --------------------------------------------------------------------------------------------
// Action: DOWNLOAD (Supports HTTP Range Requests & Streaming Proxy Bypass)
// --------------------------------------------------------------------------------------------
if ($action === 'download') {
    $model = isset($_GET['model']) ? trim($_GET['model']) : '';
    $type = (isset($_GET['type']) && $_GET['type'] === 'best') ? 'best' : 'latest';
    $target_filename = get_model_filename($model, $type);
    $target_file = $storage_dir . '/' . $target_filename;

    if (!file_exists($target_file)) {
        if ($model === 'gemma2cerebro' && file_exists($storage_dir . '/gemma2cerebro_latest.pt')) {
            $target_file = $storage_dir . '/gemma2cerebro_latest.pt';
        } else {
            // Fallback to legacy un-prefixed file if model-specific file does not exist yet
            $fallback = $storage_dir . '/' . $type . '.pt';
            if (file_exists($fallback)) {
                $target_file = $fallback;
            } else {
                http_response_code(404);
                header('Content-Type: application/json');
                echo json_encode(['error' => "Checkpoint {$target_filename} not found on server"]);
                exit;
            }
        }
    }

    $file_size = filesize($target_file);

    // Disable all active PHP output buffering to avoid server memory ballooning & throttling
    while (ob_get_level()) {
        ob_end_clean();
    }

    // Inform reverse proxies and Cloudflare not to buffer response
    header('X-Accel-Buffering: no');
    header('Accept-Ranges: bytes');
    header('Content-Description: File Transfer');
    header('Content-Type: application/octet-stream');
    header('Content-Disposition: attachment; filename="' . basename($target_file) . '"');
    header('Cache-Control: no-cache, no-store, must-revalidate');
    header('Pragma: no-cache');

    $start = 0;
    $end = $file_size - 1;

    // Check for HTTP Range header
    if (isset($_SERVER['HTTP_RANGE'])) {
        if (preg_match('/bytes=\h*(\d+)-(\d*)[\D.*]?/i', $_SERVER['HTTP_RANGE'], $matches)) {
            $start = intval($matches[1]);
            if (!empty($matches[2])) {
                $end = min(intval($matches[2]), $file_size - 1);
            }
            http_response_code(206);
            header("Content-Range: bytes {$start}-{$end}/{$file_size}");
        }
    }

    $length = $end - $start + 1;
    header('Content-Length: ' . $length);

    $handle = fopen($target_file, 'rb');
    if ($handle === false) {
        http_response_code(500);
        exit;
    }

    if ($start > 0) {
        fseek($handle, $start);
    }

    $bytes_remaining = $length;
    while (!feof($handle) && $bytes_remaining > 0) {
        $read_size = min(2 * 1024 * 1024, $bytes_remaining);
        $buffer = fread($handle, $read_size);
        if ($buffer === false || strlen($buffer) === 0) {
            break;
        }
        echo $buffer;
        flush();
        $bytes_remaining -= strlen($buffer);
    }
    fclose($handle);
    exit;
}

// --------------------------------------------------------------------------------------------
// Action: UPLOAD_CHUNK (Cloudflare 100 MB Safe, Atomic Overwrite)
// --------------------------------------------------------------------------------------------
if ($action === 'upload_chunk') {
    header('Content-Type: application/json');

    $model = isset($_POST['model']) ? trim($_POST['model']) : (isset($_GET['model']) ? trim($_GET['model']) : '');
    $type = (isset($_POST['type']) && $_POST['type'] === 'best') ? 'best' : 'latest';
    $target_filename = get_model_filename($model, $type);

    $chunk_index = isset($_POST['chunk_index']) ? intval($_POST['chunk_index']) : 0;
    $total_chunks = isset($_POST['total_chunks']) ? intval($_POST['total_chunks']) : 1;
    $step = isset($_POST['step']) ? intval($_POST['step']) : 0;
    $loss = isset($_POST['loss']) ? floatval($_POST['loss']) : 0.0;
    $source = isset($_POST['source']) ? preg_replace('/[^a-zA-Z0-9_\-]/', '', $_POST['source']) : 'unknown';

    $upload_id = isset($_POST['upload_id']) ? preg_replace('/[^a-zA-Z0-9_\-]/', '', $_POST['upload_id']) : (isset($_GET['upload_id']) ? preg_replace('/[^a-zA-Z0-9_\-]/', '', $_GET['upload_id']) : '');
    $part_file = !empty($upload_id) ? ($storage_dir . '/' . $target_filename . '.' . $upload_id . '.part') : ($storage_dir . '/' . $target_filename . '.part');
    $final_file = $storage_dir . '/' . $target_filename;

    // Retrieve binary chunk payload
    $chunk_data = null;
    if (isset($_FILES['chunk']) && is_uploaded_file($_FILES['chunk']['tmp_name'])) {
        $chunk_data = file_get_contents($_FILES['chunk']['tmp_name']);
    } else {
        $chunk_data = file_get_contents('php://input');
    }

    if ($chunk_data === null || strlen($chunk_data) === 0) {
        http_response_code(400);
        echo json_encode(['error' => 'Empty chunk payload received']);
        exit;
    }

    // Guard against out-of-order chunks
    if ($chunk_index > 0 && !file_exists($part_file)) {
        http_response_code(400);
        echo json_encode(['error' => "Out-of-order chunk: cannot append chunk {$chunk_index} without existing part file"]);
        exit;
    }

    // First chunk creates / truncates; subsequent chunks append
    $mode = ($chunk_index === 0) ? 'wb' : 'ab';
    $fp = fopen($part_file, $mode);
    if (!$fp) {
        http_response_code(500);
        echo json_encode(['error' => 'Unable to open file for writing']);
        exit;
    }
    fwrite($fp, $chunk_data);
    fclose($fp);

    // If final chunk, atomically replace the target checkpoint
    if ($chunk_index === $total_chunks - 1) {
        // Integrity check: verify zip archive is not truncated/corrupted before replacing target
        $is_valid_zip = false;
        if (class_exists('ZipArchive')) {
            $za = new ZipArchive();
            $res = $za->open($part_file, ZipArchive::CHECKCONS);
            if ($res === true) {
                $is_valid_zip = true;
                $za->close();
            }
        } else {
            $fh = fopen($part_file, 'rb');
            if ($fh) {
                $head = fread($fh, 4);
                fseek($fh, -65536, SEEK_END);
                $tail = fread($fh, 65536);
                fclose($fh);
                if ($head === "PK\x03\x04" && strpos($tail, "PK\x05\x06") !== false) {
                    $is_valid_zip = true;
                }
            }
        }
        if (!$is_valid_zip) {
            @unlink($part_file);
            http_response_code(400);
            echo json_encode(['error' => 'Corrupted checkpoint upload: failed zip integrity check']);
            exit;
        }

        if (!rename($part_file, $final_file)) {
            http_response_code(500);
            echo json_encode(['error' => 'Failed to finalize atomic checkpoint write']);
            exit;
        }

        // Update manifest
        $manifest_file = $storage_dir . '/manifest.json';
        $manifest = file_exists($manifest_file) ? json_decode(file_get_contents($manifest_file), true) : [];
        $norm_m = normalize_model_name($model);
        $manifest_key = !empty($norm_m) ? ($norm_m . '_' . $type) : $type;
        $manifest[$manifest_key] = [
            'model' => $norm_m,
            'type' => $type,
            'step' => $step,
            'loss' => $loss,
            'source' => $source,
            'updated_at' => date('c'),
            'size_bytes' => filesize($final_file),
            'md5' => md5_file($final_file)
        ];

        // Auto-promote: If a newer 'best' checkpoint has step >= current 'latest', mirror it to 'latest' at 0 network cost
        if ($type === 'best') {
            $latest_filename = get_model_filename($model, 'latest');
            $latest_file = $storage_dir . '/' . $latest_filename;
            $latest_key = !empty($norm_m) ? ($norm_m . '_latest') : 'latest';
            $curr_latest_step = isset($manifest[$latest_key]['step']) ? intval($manifest[$latest_key]['step']) : 0;
            if ($step >= $curr_latest_step) {
                @copy($final_file, $latest_file);
                $manifest[$latest_key] = [
                    'model' => $norm_m,
                    'type' => 'latest',
                    'step' => $step,
                    'loss' => $loss,
                    'source' => $source,
                    'updated_at' => date('c'),
                    'size_bytes' => filesize($final_file),
                    'md5' => $manifest[$manifest_key]['md5']
                ];
            }
        }
        file_put_contents($manifest_file, json_encode($manifest, JSON_PRETTY_PRINT));

        // Auto-record completed checkpoint into model loss history
        if (!empty($norm_m) && $step > 0 && $loss > 0) {
            $h_file = $storage_dir . '/loss_history_' . $norm_m . '.json';
            $h_list = (file_exists($h_file) && filesize($h_file) > 2) ? (json_decode(file_get_contents($h_file), true) ?: []) : [];
            $h_by_step = [];
            foreach ($h_list as $item) {
                if (isset($item['step'])) $h_by_step[intval($item['step'])] = $item;
            }
            $h_by_step[intval($step)] = [
                'step' => intval($step),
                'loss' => floatval($loss),
                'eval_loss' => isset($h_by_step[intval($step)]['eval_loss']) ? $h_by_step[intval($step)]['eval_loss'] : null,
                'status' => 'SMOOTH_DESCENT',
                'timestamp' => date('c'),
                'source' => $source,
                'md5' => md5_file($final_file),
                'size_bytes' => filesize($final_file),
            ];
            ksort($h_by_step);
            $h_clean = array_values($h_by_step);
            if (count($h_clean) > 10000) $h_clean = array_slice($h_clean, -10000);
            file_put_contents($h_file, json_encode($h_clean, JSON_PRETTY_PRINT));
        }

        echo json_encode([
            'success' => true,
            'status' => 'completed',
            'model' => $norm_m,
            'type' => $type,
            'chunk_index' => $chunk_index,
            'total_chunks' => $total_chunks,
            'manifest' => $manifest[$manifest_key]
        ]);
        exit;
    }

    echo json_encode([
        'success' => true,
        'status' => 'in_progress',
        'model' => normalize_model_name($model),
        'type' => $type,
        'chunk_index' => $chunk_index,
        'total_chunks' => $total_chunks
    ]);
    exit;
}

// Action: CLEAR / RESET (Clean historical loss histories and checkpoints)
// --------------------------------------------------------------------------------------------
if ($action === 'clear' || $action === 'reset') {
    header('Content-Type: application/json');

    $model = isset($_REQUEST['model']) ? normalize_model_name($_REQUEST['model']) : '';
    $clear_all = isset($_REQUEST['all']) && ($_REQUEST['all'] === 'true' || $_REQUEST['all'] === '1');
    $clear_checkpoints = isset($_REQUEST['checkpoints']) && ($_REQUEST['checkpoints'] === 'true' || $_REQUEST['checkpoints'] === '1');

    $deleted_files = [];
    if (!empty($model) && !$clear_all) {
        $targets = [
            $storage_dir . '/token_tracker_' . $model . '.json',
            $storage_dir . '/loss_history_' . $model . '.json',
            $storage_dir . '/active_session_' . $model . '.json',
        ];
        if ($clear_checkpoints) {
            $targets[] = $storage_dir . '/' . $model . '_latest.pt';
            $targets[] = $storage_dir . '/' . $model . '_best.pt';
        }
        foreach ($targets as $f) {
            if (file_exists($f)) {
                unlink($f);
                $deleted_files[] = basename($f);
            }
        }
        $manifest_file = $storage_dir . '/manifest.json';
        if (file_exists($manifest_file)) {
            $manifest = json_decode(file_get_contents($manifest_file), true);
            if (is_array($manifest)) {
                unset($manifest[$model . '_latest']);
                unset($manifest[$model . '_best']);
                unset($manifest['active_session_' . $model]);
                file_put_contents($manifest_file, json_encode($manifest, JSON_PRETTY_PRINT));
            }
        }
    }

    echo json_encode([
        'success' => true,
        'status' => 'cleared',
        'model' => $model ?: 'all',
        'deleted' => $deleted_files,
        'server_time' => date('c'),
    ]);
    exit;
}

http_response_code(400);
header('Content-Type: application/json');
echo json_encode(['error' => "Unknown action '{$action}'"]);
