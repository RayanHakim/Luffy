import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import cv2
import mediapipe as mp
import numpy as np
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

mp_hands = mp.solutions.hands
# Gunakan 2 tangan (Satu tangan target, satu tangan penarik)
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

cap = cv2.VideoCapture(0)

# State global untuk mengunci target tarikan
sedang_menarik = False
snapshot_frame = None
info_jari_target = None  # Menyimpan koordinat (pangkal, ujung_asli)

# Daftar indeks ujung jari dan pangkalnya di MediaPipe
# Jempol (4->2), Telunjuk (8->5), Tengah (12->9), Manis (16->13), Kelingking (20->17)
PASANGAN_JARI = [(4, 2), (8, 5), (12, 9), (16, 13), (20, 17)]

print("\n=======================================================")
print("   LUFFY PROGRAM: ALL FINGERS + SMOOTH ELASTIC DISTORT ")
print("=======================================================")
print("CARA PAKAI:")
print("1. Angkat tangan kiri (buka semua jari).")
print("2. Gunakan jempol & telunjuk tangan kanan untuk MENCUBIT")
print("   salah satu ujung jari tangan kiri (bisa telunjuk, tengah, dll).")
print("3. Tarik! Jari tersebut akan melar halus mengikuti cubitan.")
print("Tekan 'q' untuk keluar.\n")

while cap.isOpened():
    success, frame = cap.read()
    if not success: break
    
    frame = cv2.flip(frame, 1)
    h, w, c = frame.shape
    
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)
    
    posisi_pencubit = None
    kandidat_target = [] # Menampung semua ujung jari yang tersedia di layar
    
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # 1. Cari tahu apakah tangan ini sedang mencubit (Tangan Penarik)
            c_thumb = hand_landmarks.landmark[4]
            c_index = hand_landmarks.landmark[8]
            cx_t, cy_t = int(c_thumb.x * w), int(c_thumb.y * h)
            cx_i, cy_i = int(c_index.x * w), int(c_index.y * h)
            
            jarak_cubit = np.hypot(cx_i - cx_t, cy_i - cy_t)
            
            # Jika merapatkan jempol & telunjuk, kunci sebagai posisi pencubit
            if jarak_cubit < 35:
                posisi_pencubit = (int((cx_i + cx_t)/2), int((cy_i + cy_t)/2))
            
            # 2. Ambil semua koordinat ujung dan pangkal jari yang ada di layar
            for ujung, pangkal in PASANGAN_JARI:
                pt_ujung = hand_landmarks.landmark[ujung]
                pt_pangkal = hand_landmarks.landmark[pangkal]
                
                pos_ujung = (int(pt_ujung.x * w), int(pt_ujung.y * h))
                pos_pangkal = (int(pt_pangkal.x * w), int(pt_pangkal.y * h))
                
                kandidat_target.append({
                    'ujung': pos_ujung,
                    'pangkal': pos_pangkal
                })

        # LOGIKA DETEKSI JARI MANA YANG DICUBIT
        if posisi_pencubit and not sedang_menarik:
            # Cari jari mana yang jarak ujungnya paling dekat dengan cubitan tangan kanan
            for jari in kandidat_target:
                jarak_ke_cubitan = np.hypot(posisi_pencubit[0] - jari['ujung'][0], 
                                            posisi_pencubit[1] - jari['ujung'][1])
                
                # Jika jaraknya di bawah 45 piksel, berarti jari ini yang sedang dicubit!
                if jarak_ke_cubitan < 45:
                    # Cek validitas pangkal agar tidak mencubit tangannya sendiri
                    if np.hypot(posisi_pencubit[0] - jari['pangkal'][0], posisi_pencubit[1] - jari['pangkal'][1]) > 40:
                        sedang_menarik = True
                        snapshot_frame = frame.copy() # Ambil snapshot tekstur saat itu
                        info_jari_target = {
                            'pangkal': jari['pangkal'],
                            'ujung_asli': jari['ujung']
                        }
                        break

        # Lepas cubitan jika tangan penarik hilang dari kamera
        if sedang_menarik and not posisi_pencubit:
            sedang_menarik = False
            snapshot_frame = None
            info_jari_target = None

        # LOGIKA PROSES PELEMBUTAN (SMOOTH TRANSFORMATION)
        if sedang_menarik and posisi_pencubit and snapshot_frame is not None:
            p_x, p_y = info_jari_target['pangkal']
            u_asli_x, u_asli_y = info_jari_target['ujung_asli']
            u_baru_x, u_baru_y = posisi_pencubit
            
            # Lebar area jari yang mau ditarik (agar daging kiri kanan jari ikut terbawa secara proporsional)
            r = 25 
            
            # Hitung vektor arah jari asli
            v_x, v_y = u_asli_x - p_x, u_asli_y - p_y
            len_v = np.hypot(v_x, v_y)
            
            if len_v > 0:
                # Cari koordinat tegak lurus (perpendicular) untuk membuat bidang kotak jari asli
                dx, dy = -v_y / len_v * r, v_x / len_v * r
                
                # 4 Titik Sudut Bidang Jari Asli (Trapesium/Kotak asal)
                src_pts = np.float32([
                    [p_x + dx, p_y + dy],       # Pangkal Kiri
                    [p_x - dx, p_y - dy],       # Pangkal Kanan
                    [u_asli_x - dx, u_asli_y - dy], # Ujung Kanan Asli
                    [u_asli_x + dx, u_asli_y + dy]  # Ujung Kiri Asli
                ])
                
                # Hitung vektor arah jari yang baru (yang melar ditarik)
                v_new_x, v_new_y = u_baru_x - p_x, u_baru_y - p_y
                len_v_new = np.hypot(v_new_x, v_new_y)
                
                if len_v_new > 0:
                    dx_n, dy_n = -v_new_y / len_v_new * r, v_new_x / len_v_new * r
                    
                    # 4 Titik Sudut Bidang Jari Baru (Meregang mengikuti cubitan secara mulus)
                    dst_pts = np.float32([
                        [p_x + dx_n, p_y + dy_n],
                        [p_x - dx_n, p_y - dy_n],
                        [u_baru_x - dx_n, u_baru_y - dy_n],
                        [u_baru_x + dx_n, u_baru_y + dy_n]
                    ])
                    
                    # Lakukan Perspective Warp (Melenturkan tekstur gambar asli secara geometris)
                    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
                    warped = cv2.warpPerspective(snapshot_frame, M, (w, h))
                    
                    # Buat Masking halus agar potongan jari melar menyatu tanpa border patah
                    mask = np.zeros((h, w), dtype=np.uint8)
                    cv2.fillConvexPoly(mask, np.int32(dst_pts), 255)
                    mask = cv2.GaussianBlur(mask, (11, 11), 0) # Efek blur di tepian agar super smooth
                    
                    # Gabungkan gambar kamera real-time dengan jari karet hasil warping
                    mask_idx = mask > 0
                    frame[mask_idx] = cv2.addWeighted(warped, 1.0, frame, 0.0, 0)[mask_idx]

            cv2.putText(frame, "GOMU GOMU NO...", (30, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 165, 255), 2)

    cv2.imshow("Luffy - All Fingers Smooth Elastic", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()