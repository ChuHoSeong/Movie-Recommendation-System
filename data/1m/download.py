import os
import ssl
import urllib.request
import zipfile

# SSL 인증서 검증 우회 (Windows 환경 대응)
ssl._create_default_https_context = ssl._create_unverified_context

URL = "https://files.grouplens.org/datasets/movielens/ml-1m.zip"
RAW_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "raw")


def download_1m():
    os.makedirs(RAW_DIR, exist_ok=True)
    zip_path = os.path.join(RAW_DIR, "ml-1m.zip")

    print("[1/2] 1M 데이터 다운로드 중...")
    urllib.request.urlretrieve(URL, zip_path)

    print("[2/2] 압축 해제 중...")
    with zipfile.ZipFile(zip_path, "r") as z:
        for member in z.infolist():
            filename = os.path.basename(member.filename)
            if filename:
                with z.open(member) as src, open(os.path.join(RAW_DIR, filename), "wb") as dst:
                    dst.write(src.read())

    os.remove(zip_path)
    print(f"[완료] {RAW_DIR} 에 저장 완료")


if __name__ == "__main__":
    download_1m()
