# -*- mode: python ; coding: utf-8 -*-


a = Analysis(
    ['GUI.py'],
    pathex=[],
    binaries=[],
    datas=[],
    hiddenimports=['tkinter', 'screeninfo', 'customtkinter', 'tkintermapview', 'PIL', 'pandas','sqlalchemy','pymysql','mysql','mysqlclient', "mysql.connector"],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    icon=['assets\\icon.png'],
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    Tree('..\\ACE-TI\\assets', prefix='assets\\'),
    [],
    name='ACETI-GUI',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=['assets\\icon_aceti.png'],
)
