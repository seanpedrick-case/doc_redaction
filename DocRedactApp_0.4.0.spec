# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_data_files
from PyInstaller.utils.hooks import collect_all

datas = [('tesseract/', 'tesseract/'), ('poppler/poppler-24.02.0/', 'poppler/poppler-24.02.0/')]
binaries = []
hiddenimports = ['gradio_image_annotation', 'pyarrow.vendored.version', 'pydicom.encoders', 'safehttpx', 'presidio_analyzer', 'presidio_anonymizer', 'presidio_image_redactor']
datas += collect_data_files('gradio_client')
datas += collect_data_files('gradio')
datas += collect_data_files('gradio_image_annotation')
tmp_ret = collect_all('gradio_image_annotation')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]
tmp_ret = collect_all('safehttpx')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]
tmp_ret = collect_all('presidio_analyzer')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]
tmp_ret = collect_all('presidio_anonymizer')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]
tmp_ret = collect_all('presidio_image_redactor')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]


a = Analysis(
    ['app.py'],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=['build_deps'],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
    module_collection_mode={
        'gradio': 'py',  # Collect gradio package as source .py files
    }
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='DocRedactApp_0.4.0',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='DocRedactApp_0.4.0',
)
