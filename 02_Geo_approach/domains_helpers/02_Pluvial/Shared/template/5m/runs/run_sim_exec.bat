set TUFLOWEXE=c:\TUFLOW\Releases\{{tu_version}}\TUFLOW_iSP_w64.exe
set RUN=start "TUFLOW" /wait "%TUFLOWEXE%" -b -acf -x

REM set RUN=start "TUFLOW" /wait "%TUFLOWEXE%" -b -acf -t -nlc -nt14

%RUN% -s1 s0 -s2 e24 -s3 G00 -s4 D01 -s5 o24 -s6 ms5 -s7 Hs -e1 I11 -e2 d1440 -e3 r00005 {{ domain }}.tcf
rem %RUN% -s1 s0 -s2 e24 -s3 G00 -s4 D01 -s5 o24 -s6 ms5 -s7 Hs -e1 I11 -e2 d1440 -e3 r00020 {{ domain }}.tcf
REM %RUN% -s1 s0 -s2 e24 -s3 G00 -s4 D01 -s5 o24 -s6 ms5 -s7 Hs -e1 I51 -e2 d1440 -e3 r00050 {{ domain }}.tcf
rem %RUN% -s1 s0 -s2 e24 -s3 G00 -s4 D01 -s5 o24 -s6 ms5 -s7 Hs -e1 I51 -e2 d1440 -e3 r00100 {{ domain }}.tcf
REM %RUN% -s1 s0 -s2 e24 -s3 G00 -s4 D01 -s5 o24 -s6 ms5 -s7 Hs -e1 I02 -e2 d1440 -e3 r00200 {{ domain }}.tcf
rem %RUN% -s1 s0 -s2 e24 -s3 G00 -s4 D01 -s5 o24 -s6 ms5 -s7 Hs -e1 I02 -e2 d1440 -e3 r00500 {{ domain }}.tcf
rem %RUN% -s1 s0 -s2 e24 -s3 G00 -s4 D01 -s5 o24 -s6 ms5 -s7 Hs -e1 I02 -e2 d1440 -e3 r01000 {{ domain }}.tcf
rem %RUN% -s1 s0 -s2 e24 -s3 G00 -s4 D01 -s5 o24 -s6 ms5 -s7 Hs -e1 I00 -e2 d1440 -e3 r10000 {{ domain }}.tcf
rem pause