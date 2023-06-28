ECHO ON
set root=C:\Users\%username%\anaconda3\
call %root%\Scripts\activate.bat %root%
start ""  http://127.0.0.1:8000/
call conda activate bln_check

call python BLanc_de_Noir_Check.py

@ECHO ---------------------------------------------------------------------
@ECHO ---------------------------------------------------------------------

pause
