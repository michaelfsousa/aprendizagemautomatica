# -*- mode: python -*-

block_cipher = None


a = Analysis(['start.py'],
             pathex=['D:\\FORMA��O ACAD�MICA\\UFC - Mestrado Ci�ncia da Computa��o\\Disserta��o\\Materiad de pesquisa\\Projeto teclado de gestos'],
             binaries=[],
             datas=[],
             hiddenimports=[],
             hookspath=[],
             runtime_hooks=[],
             excludes=[],
             win_no_prefer_redirects=False,
             win_private_assemblies=False,
             cipher=block_cipher,
             noarchive=False)
pyz = PYZ(a.pure, a.zipped_data,
             cipher=block_cipher)
exe = EXE(pyz,
          a.scripts,
          a.binaries,
          a.zipfiles,
          a.datas,
          [],
          name='start',
          debug=False,
          bootloader_ignore_signals=False,
          strip=False,
          upx=True,
          runtime_tmpdir=None,
          console=True )
