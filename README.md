#Simulator Arm FVP memerlukan library Python 3.9 (libpython3.9.so.1.0). Karena Linux Mint modern biasanya membawa versi Python yang lebih baru, kita harus menambahkan repositori khusus (deadsnakes) untuk menginstalnya.

## Tambahkan repositori PPA deadsnakes untuk Python versi lama
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt update

## Instal libpython3.9 yang dibutuhkan oleh simulator FVP
sudo apt install libpython3.9

## Instal picocom untuk memonitor port serial board fisik nanti
sudo apt install picocom

## PENTING:harus log out dan log in kembali agar perubahan grup ini aktif!
sudo usermod -aG dialout $USER

# Pembuatan environment dan instalasi Vela
## Masuk ke direktori kerja utama
cd ~/tinyml-arm-workflow

## Buat dan aktifkan venv lokal
python3 -m venv .venv
source .venv/bin/activate

## Instal compiler Vela (untuk optimasi model ke Ethos-U55 NPU)
pip install ethos-u-vela

## Verifikasi instalasi vela
vela --version

# Install Arm FVP
## Ekstrak paket installer FVP Corstone-300
tar -xzf FVP_Corstone_SSE-300_*.tgz

## Jalankan script install dan arahkan ke ~/FVP_Corstone_SSE-300
./FVP_Corstone_SSE-300.sh

## Tambahkan binary FVP ke PATH sistem melalui ~/.bashrc
echo 'export PATH=$PATH:$HOME/FVP_Corstone_SSE-300/models/Linux64_GCC-9.3' >> ~/.bashrc
source ~/.bashrc
