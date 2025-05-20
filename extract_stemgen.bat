@echo off
:: converts mp3 to wav & calls stemgen
set "FILENAME=%~1"
if not exist ".\temp" mkdir ".\temp"
ffmpeg -i "D:\music\%FILENAME%.mp3" ".\temp\%FILENAME%.wav"
python "D:\repos\stemgen\stemgen.py" --i ".\temp\%FILENAME%.wav" -f aac -o "D:\music_prod\stems_openfmt" -n htdemucs
del ".\temp\%FILENAME%.wav"
powershell -c "[console]::beep(1000,500)"
