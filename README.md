# Luffy - Gomu Gomu Finger Stretch

Aplikasi Python sederhana untuk mensimulasikan kekuatan karet Luffy dari anime One Piece secara *real-time* menggunakan kamera laptop. Program ini mendeteksi cubitan pada ujung jari mana pun (jempol, telunjuk, tengah, manis, kelingking) lalu menarik tekstur kulit jari tersebut secara elastis dan halus.

## 🚀 Fitur
- **Deteksi Multi-Jari:** Bisa menarik semua jari (bukan cuma telunjuk).
- **Efek Melar Halus:** Menggunakan teknik *Perspective Transformation* dan *Blur Blending* agar hasil tarikan tidak patah/kotak-kotak.
- **Kamera Real-time:** Latar belakang kamera tetap berjalan normal tanpa membekukan layar (*no freeze*).

## 🛠️ Prasyarat
Sebelum menjalankan, pastikan laptop kamu sudah terinstal Python 3.12,kalau Python 3.13 gak bisa dan library berikut:

```bash
pip install opencv-python mediapipe numpy
