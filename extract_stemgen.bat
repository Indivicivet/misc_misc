@echo off
:: converts mp3 to wav & calls stemgen
set "FILENAME=%~1"

if not exist ".\temp" mkdir ".\temp"

ffmpeg -i "D:\music\%FILENAME%.mp3" ".\temp\%FILENAME%.wav"

call "D:\music_prod\demucs_py310_v2\Scripts\python.exe" "D:\repos\stemgen\stemgen.py" --i ".\temp\%FILENAME%.wav" -f aac -o "D:\music_prod\stems_openfmt"

del ".\temp\%FILENAME%.wav"

powershell -c "[console]::beep(1000,500)"
