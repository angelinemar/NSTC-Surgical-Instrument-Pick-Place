# Audit topic training dan benchmark grid P4

Diperbarui: 2026-09-17T23:47:50

## Kesimpulan sementara

Schema lengkap menyediakan bahan untuk perception/object recognition dan diffusion policy. Itu tidak membuktikan dataset cukup beragam atau semua pose robot berhasil. Benchmark fisik dan audit topic dinilai terpisah.

Inventaris awal: 115 H5, 57 pasangan split; 98 file lolos audit topic saat ini, 37 pasangan lolos audit topic kedua segmen dan pemeriksaan metadata QC fisik.

## Kebutuhan model

| Kebutuhan | Topic / kontrak | Status dan penggunaan |
| --- | --- | --- |
| Input gambar | 6 RGB uint8, T x 224 x 224 x 3 | Pilih view yang relevan; wrist alias bukan kamera tambahan |
| Label recognition | 6 semantic uint16, T x 224 x 224 | ID 0 background, 1 robot, 2 tray, 3..7 instrumen |
| RGB-D geometry | depth + intrinsics_output + position_b + quaternion_b_ros | Intrinsics sesuai crop; depth Z dari distance_to_image_plane, bukan jarak radial |
| Propriosepsi | robot_proprio T x 16 / state T x 18 | 7 joints, 2 fingers, EE XYZ+quat; state menambah object dan skill ID |
| Target policy | actions T x 8 | absolute robot-base XYZ, quaternion wxyz, gripper -1/+1 |
| Kondisi task | object_type_id 0..4; skill_id pick=0/place=1 | Bukan label class mask 3..7 |
| Supervision tambahan | conditioning_mask_*; teacher_grasp_pose_b | Mask pick untuk instrumen; mask place untuk tray. Teacher grasp place sengaja NaN |
| Timing | Urutan baris; step_ids lokal tahap | File lama tidak punya control_dt_s eksplisit; verifikasi dt dari run/config sebelum konversi |

Jangan memasukkan debug_gt, semantic GT, supervision_gt, atau stage_id expert ke input policy yang tidak tersedia saat deployment. Semantic dan supervision sah sebagai label loss perception. Pipeline aktif mengecualikan ID target simulator dan state18; input hanya RGB + robot_proprio. Exporter mensyaratkan satu instrumen actionable; perintah target eksplisit belum diaktifkan.

## Kompatibilitas training

`C:/IsaacLab/configs_scissor_dp.json` yang diperiksa hanya memakai low_dim=[state], rgb=[], depth=[]. Konfigurasi lama ini tidak melatih image-conditioned diffusion policy. H5 P4 juga perlu adapter/export ke struktur yang diminta loader training, bukan hanya mengganti path dataset.

Audit menemukan 68 pergantian tanda quaternion antarsample. q dan -q menyatakan orientasi yang sama; samakan tanda secara berurutan saat export, lalu normalisasi quaternion hasil policy sebelum kontrol. Dimensi dan arti action recorder tetap 8D.

6 file memiliki RGB datar pada seluruh kamera. Data gripper kontinu yang gagal kontrak policy biner belum tentu tidak berguna bagi training perception: evaluasi kelayakan perception dari RGB/semantic, dan kelayakan policy dari action/propriosepsi/QC fisik. Jangan memakai status target object sebagai satu-satunya label kelas untuk gambar multiobjek.

Pisahkan train/validation/test berdasarkan episode/session; pasangan pick dan place harus berada pada split yang sama. Jangan random-split frame bertetangga. Hitung statistik normalisasi dari training split saja, tangani kontinuitas tanda quaternion, dan jangan membuat action chunk melintasi batas episode atau batas policy pick/place. Jumlah data ini belum membuktikan generalisasi; lakukan evaluasi pose/grid/yaw dan visual yang ditahan dari training.

## Inventaris dan coverage historis

| Instrumen | Pasangan bersih topic + metadata QC | Grid tersimpan historis (ID 0..9) |
| --- | ---: | --- |
| scalpel | 8 | 0, 1, 2, 3, 5 |
| scissor | 12 | 0, 1, 2, 3 |
| love_retractor | 8 | 1, 6, 7 |
| kelly | 4 | 0, 5, 8 |
| scalpel_type2 | 5 | 1, 3 |

Angka historis mencampur beberapa versi recorder dan tidak boleh dianggap coverage controller aktif. Controller metadata inventaris awal: {'missing': 113, 'tcp_frame_correct_bounded_v4': 2}. Tidak ada instrumen yang memiliki bukti historis lengkap seluruh 10 grid dalam inventaris ini.

## Benchmark controller aktif

Rencana: 50 kasus; yaw [0.0]. Maksimal 2 attempts per kombinasi, default wrist awal, tray berisi empat objek lain. Kegagalan tetap dicatat; hanya pasangan sukses lolos audit yang disimpan. Ini belum menyapu semua titik dalam cell, jitter, roll scalpel, atau occupancy.

Selesai 50/50; hasil {'PASS': 50}; aktif None. Semua kasus selesai: True.

Sukses langsung: 48; sukses setelah retry: 2.

Tiap cell ditulis PASS/FAIL/ERROR/PENDING sesuai jumlah kasus dalam rencana. PASS memerlukan pasangan H5 dan audit. PENDING bukan sukses.

| Instrumen | Grid 0 | Grid 1 | Grid 2 | Grid 3 | Grid 4 | Grid 5 | Grid 6 | Grid 7 | Grid 8 | Grid 9 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| scalpel | 1/0/0/0 | 1/0/0/0 | 1/0/0/0 | 1/0/0/0 | 1/0/0/0 | 1/0/0/0 | 1/0/0/0 | 1/0/0/0 | 1/0/0/0 | 1/0/0/0 |
| scissor | 1/0/0/0 | 1/0/0/0 | 1/0/0/0 | 1/0/0/0 | 1/0/0/0 | 1/0/0/0 | 1/0/0/0 | 1/0/0/0 | 1/0/0/0 | 1/0/0/0 |
| love_retractor | 1/0/0/0 | 1/0/0/0 | 1/0/0/0 | 1/0/0/0 | 1/0/0/0 | 1/0/0/0 | 1/0/0/0 | 1/0/0/0 | 1/0/0/0 | 1/0/0/0 |
| kelly | 1/0/0/0 | 1/0/0/0 | 1/0/0/0 | 1/0/0/0 | 1/0/0/0 | 1/0/0/0 | 1/0/0/0 | 1/0/0/0 | 1/0/0/0 | 1/0/0/0 |
| scalpel_type2 | 1/0/0/0 | 1/0/0/0 | 1/0/0/0 | 1/0/0/0 | 1/0/0/0 | 1/0/0/0 | 1/0/0/0 | 1/0/0/0 | 1/0/0/0 | 1/0/0/0 |

Detail machine-readable: `debug/test_runs/benchmark_grid_baseline_20260917_v11/summary.json`. Jangan mengubah controller/layout selama benchmark; fingerprint sumber dicek antar-kasus.

## Audit seluruh topic

Catalog berikut mencakup seluruh 76 topic schema scissor v14. T adalah panjang segmen; pick dan place tidak harus mempunyai T yang sama. Semua schema historis, nilai nonfinite, RGB, depth, semantic, alias dan kalibrasi ada pada `validation/training_topic_audit_20260917/file_*.json`.

| Topic | Shape contoh / dtype | Peran |
| --- | --- | --- |
| `actions` | [244, 8] / float32 | Target diffusion policy: absolute XYZ + quaternion wxyz + gripper |
| `camera_calibration/cam_left/intrinsics_output` | [244, 3, 3] / float32 | Kalibrasi geometri RGB-D; bukan label kelas |
| `camera_calibration/cam_left/intrinsics_raw` | [244, 3, 3] / float32 | Kalibrasi geometri RGB-D; bukan label kelas |
| `camera_calibration/cam_left/position_b` | [244, 3] / float32 | Kalibrasi geometri RGB-D; bukan label kelas |
| `camera_calibration/cam_left/quaternion_b_ros` | [244, 4] / float32 | Kalibrasi geometri RGB-D; bukan label kelas |
| `camera_calibration/cam_right/intrinsics_output` | [244, 3, 3] / float32 | Kalibrasi geometri RGB-D; bukan label kelas |
| `camera_calibration/cam_right/intrinsics_raw` | [244, 3, 3] / float32 | Kalibrasi geometri RGB-D; bukan label kelas |
| `camera_calibration/cam_right/position_b` | [244, 3] / float32 | Kalibrasi geometri RGB-D; bukan label kelas |
| `camera_calibration/cam_right/quaternion_b_ros` | [244, 4] / float32 | Kalibrasi geometri RGB-D; bukan label kelas |
| `camera_calibration/cam_top/intrinsics_output` | [244, 3, 3] / float32 | Kalibrasi geometri RGB-D; bukan label kelas |
| `camera_calibration/cam_top/intrinsics_raw` | [244, 3, 3] / float32 | Kalibrasi geometri RGB-D; bukan label kelas |
| `camera_calibration/cam_top/position_b` | [244, 3] / float32 | Kalibrasi geometri RGB-D; bukan label kelas |
| `camera_calibration/cam_top/quaternion_b_ros` | [244, 4] / float32 | Kalibrasi geometri RGB-D; bukan label kelas |
| `camera_calibration/cam_tray/intrinsics_output` | [244, 3, 3] / float32 | Kalibrasi geometri RGB-D; bukan label kelas |
| `camera_calibration/cam_tray/intrinsics_raw` | [244, 3, 3] / float32 | Kalibrasi geometri RGB-D; bukan label kelas |
| `camera_calibration/cam_tray/position_b` | [244, 3] / float32 | Kalibrasi geometri RGB-D; bukan label kelas |
| `camera_calibration/cam_tray/quaternion_b_ros` | [244, 4] / float32 | Kalibrasi geometri RGB-D; bukan label kelas |
| `camera_calibration/front/intrinsics_output` | [244, 3, 3] / float32 | Kalibrasi geometri RGB-D; bukan label kelas |
| `camera_calibration/front/intrinsics_raw` | [244, 3, 3] / float32 | Kalibrasi geometri RGB-D; bukan label kelas |
| `camera_calibration/front/position_b` | [244, 3] / float32 | Kalibrasi geometri RGB-D; bukan label kelas |
| `camera_calibration/front/quaternion_b_ros` | [244, 4] / float32 | Kalibrasi geometri RGB-D; bukan label kelas |
| `camera_calibration/grip_b/intrinsics_output` | [244, 3, 3] / float32 | Kalibrasi geometri RGB-D; bukan label kelas |
| `camera_calibration/grip_b/intrinsics_raw` | [244, 3, 3] / float32 | Kalibrasi geometri RGB-D; bukan label kelas |
| `camera_calibration/grip_b/position_b` | [244, 3] / float32 | Kalibrasi geometri RGB-D; bukan label kelas |
| `camera_calibration/grip_b/quaternion_b_ros` | [244, 4] / float32 | Kalibrasi geometri RGB-D; bukan label kelas |
| `debug_gt/legacy_state_34d` | [244, 34] / float32 | Debug simulator; jangan menjadi input policy |
| `debug_gt/object_a_pose_b` | [244, 7] / float32 | Debug simulator; jangan menjadi input policy |
| `debug_gt/scalpel_pose_b` | [244, 7] / float32 | Debug simulator; jangan menjadi input policy |
| `debug_gt/selected_object_pose_b` | [244, 7] / float32 | Debug simulator; jangan menjadi input policy |
| `debug_gt/target_slot_b` | [244, 3] / float32 | Debug simulator; jangan menjadi input policy |
| `dones` | [244] / bool | Batas episode / reward placeholder; bukan bukti sukses |
| `norm_stats/action_mean` | [8] / float32 | Statistik per episode; hitung ulang normalisasi pada training split |
| `norm_stats/action_std` | [8] / float32 | Statistik per episode; hitung ulang normalisasi pada training split |
| `norm_stats/state_mean` | [18] / float32 | Statistik per episode; hitung ulang normalisasi pada training split |
| `norm_stats/state_std` | [18] / float32 | Statistik per episode; hitung ulang normalisasi pada training split |
| `observations/cam_left_depth` | [244, 224, 224] / float16 | Input RGB-D opsional; mask depth tidak valid |
| `observations/cam_left_rgb` | [244, 224, 224, 3] / uint8 | Input visual policy/perception |
| `observations/cam_left_semantic` | [244, 224, 224] / uint16 | Label segmentasi/classification perception dari simulator |
| `observations/cam_right_depth` | [244, 224, 224] / float16 | Input RGB-D opsional; mask depth tidak valid |
| `observations/cam_right_rgb` | [244, 224, 224, 3] / uint8 | Input visual policy/perception |
| `observations/cam_right_semantic` | [244, 224, 224] / uint16 | Label segmentasi/classification perception dari simulator |
| `observations/cam_top_depth` | [244, 224, 224] / float16 | Input RGB-D opsional; mask depth tidak valid |
| `observations/cam_top_rgb` | [244, 224, 224, 3] / uint8 | Input visual policy/perception |
| `observations/cam_top_semantic` | [244, 224, 224] / uint16 | Label segmentasi/classification perception dari simulator |
| `observations/cam_tray_depth` | [244, 224, 224] / float16 | Input RGB-D opsional; mask depth tidak valid |
| `observations/cam_tray_rgb` | [244, 224, 224, 3] / uint8 | Input visual policy/perception |
| `observations/cam_tray_semantic` | [244, 224, 224] / uint16 | Label segmentasi/classification perception dari simulator |
| `observations/depth_front` | [244, 224, 224] / float16 | Alias kompatibilitas; jangan duplikasi channel |
| `observations/depth_grip_b` | [244, 224, 224] / float16 | Alias kompatibilitas; jangan duplikasi channel |
| `observations/depth_wrist` | [244, 224, 224] / float16 | Alias kompatibilitas; jangan duplikasi channel |
| `observations/front_depth` | [244, 224, 224] / float16 | Input RGB-D opsional; mask depth tidak valid |
| `observations/front_rgb` | [244, 224, 224, 3] / uint8 | Input visual policy/perception |
| `observations/front_semantic` | [244, 224, 224] / uint16 | Label segmentasi/classification perception dari simulator |
| `observations/grip_b_semantic` | [244, 224, 224] / uint16 | Label segmentasi/classification perception dari simulator |
| `observations/images_front` | [244, 224, 224, 3] / uint8 | Alias kompatibilitas; jangan duplikasi channel |
| `observations/images_grip_b` | [244, 224, 224, 3] / uint8 | Alias kompatibilitas; jangan duplikasi channel |
| `observations/images_wrist` | [244, 224, 224, 3] / uint8 | Alias kompatibilitas; jangan duplikasi channel |
| `observations/object_type_id` | [244, 1] / float32 | Metadata expert; dikeluarkan dari input model sensor-only |
| `observations/robot_proprio` | [244, 16] / float32 | Input propriosepsi robot terukur |
| `observations/segmentation_map` | [244, 224, 224] / uint16 | Alias kompatibilitas; jangan duplikasi channel |
| `observations/skill_id` | [244, 1] / float32 | Metadata expert; dikeluarkan dari input model sensor-only |
| `observations/stage_id` | [244, 1] / int32 | Analisis tahap; step_ids lokal per tahap |
| `observations/state` | [244, 18] / float32 | Legacy state + ID objek/skill; dikeluarkan dari input model |
| `observations/wrist_depth` | [244, 224, 224] / float16 | Input RGB-D opsional; mask depth tidak valid |
| `observations/wrist_rgb` | [244, 224, 224, 3] / uint8 | Input visual policy/perception |
| `rewards` | [244] / float32 | Batas episode / reward placeholder; bukan bukti sukses |
| `stage_names` | [244] / object | Analisis tahap; step_ids lokal per tahap |
| `stage_suffixes` | [244] / object | Analisis tahap; step_ids lokal per tahap |
| `step_ids` | [244] / int32 | Analisis tahap; step_ids lokal per tahap |
| `supervision_gt/conditioning_mask_cam_left` | [244, 224, 224] / uint8 | Label training perception/grasp; bukan input GT saat inference |
| `supervision_gt/conditioning_mask_cam_right` | [244, 224, 224] / uint8 | Label training perception/grasp; bukan input GT saat inference |
| `supervision_gt/conditioning_mask_cam_top` | [244, 224, 224] / uint8 | Label training perception/grasp; bukan input GT saat inference |
| `supervision_gt/conditioning_mask_cam_tray` | [244, 224, 224] / uint8 | Label training perception/grasp; bukan input GT saat inference |
| `supervision_gt/conditioning_mask_front` | [244, 224, 224] / uint8 | Label training perception/grasp; bukan input GT saat inference |
| `supervision_gt/conditioning_mask_grip_b` | [244, 224, 224] / uint8 | Label training perception/grasp; bukan input GT saat inference |
| `supervision_gt/teacher_grasp_pose_b` | [7] / float32 | Label training perception/grasp; bukan input GT saat inference |

## Masalah pada inventaris awal

Gagal kontrak topic: {"flat_rgb:cam_left": 6, "flat_rgb:cam_right": 6, "flat_rgb:cam_top": 6, "flat_rgb:cam_tray": 6, "flat_rgb:front": 6, "flat_rgb:grip_b": 6, "gripper_not_binary": 14, "nonfinite:supervision_gt/teacher_grasp_pose_b:7": 1}.

Perintah gripper kontinu pada data lama bukan kerusakan file, tetapi berbeda dari kontrak biner controller aktif. Jangan mencampurnya tanpa keputusan konversi/versi eksplisit. File yang lolos topic belum otomatis layak dimasukkan ke dataset produksi.
