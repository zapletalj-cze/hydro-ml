set TUFLOWEXE=c:\TUFLOW\Releases\2023-03-AA\TUFLOW_iSP_w64.exe
set RUN=start "TUFLOW" /wait "%TUFLOWEXE%" -b -acf -x

%RUN%  -s1 s0 -s2 e24 -s3 G01 -s4 D01 -s5 o24 -s6 os20 -s7 Hs -e1 I11 -e2 d1440 -e3 r00100 W01_0001.tcf
rem pause
