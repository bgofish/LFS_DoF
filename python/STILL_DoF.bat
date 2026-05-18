@echo off
REM Activate Anaconda base environment
call %USERPROFILE%\anaconda3\Scripts\activate.bat

REM Run your Python script
python "C:\Users\%username%\.lichtfeld\plugins\DoF\python\Still-DoF_Bokeh.py"

pause
