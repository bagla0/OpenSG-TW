@ECHO OFF
REM Minimal Sphinx build helper for the OpenSG-TW docs.
pushd %~dp0
set SOURCEDIR=.
set BUILDDIR=_build
if "%1" == "" goto html
sphinx-build -b %1 %SOURCEDIR% %BUILDDIR%\%1
goto end
:html
sphinx-build -b html %SOURCEDIR% %BUILDDIR%\html
:end
popd
