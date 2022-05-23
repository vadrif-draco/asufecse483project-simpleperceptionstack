@ECHO OFF
setlocal enableDelayedExpansion

FOR /f %%p in ('where py') do SET PYTHONPATH=%%p

if DEFINED PYTHONPATH (

    SET /p ans="Is %PYTHONPATH% the path for your preferred python version? If no, please enter your python path manually. (y/n): "

    if /i NOT "!ans!"=="y" (

        SET "PYTHONPATH="
        SET /p PYTHONPATH="Please enter your python (.exe) path manually: "

    )

    SET "ans="

) else (

    SET /p PYTHONPATH="Please enter your python (.exe) path manually: "

)

if NOT "%*"=="" (

    SET args=%*

)

if DEFINED args %PYTHONPATH% ./pipeline.py !args!
if NOT DEFINED args %PYTHONPATH% ./pipeline.py -h

SET "PYTHONPATH="
SET "args="