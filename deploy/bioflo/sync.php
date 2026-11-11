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

if ($token_provided !== $SECRET_TOKEN) {
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
        $model_status[$m] = [
            'has_latest' => $has_l,
            'has_best' => $has_b,
            'latest_size' => $has_l ? filesize($l_file) : 0,
            'best_size' => $has_b ? filesize($b_file) : 0,
            'latest_manifest' => isset($manifest[$m . '_latest']) ? $manifest[$m . '_latest'] : null,
            'best_manifest' => isset($manifest[$m . '_best']) ? $manifest[$m . '_best'] : null,
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

    $part_file = $storage_dir . '/' . $target_filename . '.part';
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
        file_put_contents($manifest_file, json_encode($manifest, JSON_PRETTY_PRINT));

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

http_response_code(400);
header('Content-Type: application/json');
echo json_encode(['error' => "Unknown action '{$action}'"]);
