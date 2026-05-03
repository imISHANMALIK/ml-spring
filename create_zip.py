import zipfile
import os

def create_zip():
    with zipfile.ZipFile('finjepa_colab_v9_dense_windows.zip', 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk('src'):
            for file in files:
                zipf.write(os.path.join(root, file))
        zipf.write('run_all.py')
        zipf.write('paper_figures.py')
        zipf.write('sweep_tau.py')

if __name__ == '__main__':
    create_zip()
    print("Zip file created successfully as finjepa_colab.zip")
