@echo off
cd /d "%~dp0"
echo === ecdlp-rho: build ===
echo.

set EXTRA=
if /I "%1"=="fermat" set EXTRA=-DINV_MODE=2
if /I "%1"=="eea" set EXTRA=-DINV_MODE=1
if not "%EXTRA%"=="" echo inversion mode: %1

where nvcc >nul 2>nul
if errorlevel 1 goto cpu

echo [1/2] Building toy solver   (rho_toy.exe)
nvcc -O3 -std=c++17 -arch=native %EXTRA% -DUSE_TOY src\rho.cu -o rho_toy.exe
if errorlevel 1 goto fail

echo [2/2] Building sample solver (rho.exe)
nvcc -O3 -std=c++17 -arch=native %EXTRA% src\rho.cu -o rho.exe
if errorlevel 1 goto fail

echo.
echo Build OK.
echo   rho_toy.exe selftest
echo   rho_toy.exe cpu 12     recovers the known toy scalar
echo   rho.exe speed          throughput test
echo   rho.exe gpu 22         sample run
goto end

:cpu
echo nvcc not found; trying MSVC.
where cl >nul 2>nul
if errorlevel 1 goto noc
cl /nologo /O2 /EHsc /TP /Fe:rho_cpu.exe src\rho.cu
if errorlevel 1 goto fail
echo Build OK (CPU). Run: rho_cpu.exe selftest
goto end

:noc
echo [!] Neither nvcc nor cl found.
echo     Install the CUDA Toolkit, or open an "x64 Native Tools Command
echo     Prompt for VS" and run this script.
goto end

:fail
echo.
echo [!] Build failed.

:end
