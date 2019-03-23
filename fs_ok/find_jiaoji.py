# -*- coding: utf-8 -*-
"""
Created on Mon Jan 28 09:49:52 2019

@author: jie
"""

lasso=['_RFCHOL1', '_AGE80', '_CHOLCH1', 'GENHLTH', '_RFBMI5', 'WTKG3',
       '_BMI5', 'DIABAGE2', 'WEIGHT2', '_BMI5CAT', '_RACEG21', '_HCVU651',
       'BPALCADV', 'CIMEMLOS', 'MARITAL', 'BPMEDADV', 'LMTJOIN3', 'TOLDHI2',
       'BPEATADV', 'MARIJANA', 'BPHI2MR', 'CHILDREN', 'EMPLOY1', 'BPEXRADV',
       '_RACE_G1', '_PHYS14D', 'DELAYMED', '_AGE_G', 'BPSLTADV', 'STRENGTH',
       'ARTTODAY', '_RFBING5', 'CHCCOPD1', 'CARERCVD', 'COPDCOGH', 'ASBIDRNK',
       'SSBFRUT3', 'RENTHOM1', 'ASTHNOW', 'DEAF', 'DIABEYE', 'COPDBTST',
       'FVGREEN1', 'JOINPAI1', 'HPVADVC2', 'EXERHMM2', 'PAFREQ1_', 'INTERNET',
       'TETANUS', 'DIABEDU', 'SEATBELT', 'RCSRLTN2', 'SDHSTRES', 'CELLFON4',
       'LADULT', 'CSRVCTL1', 'VETERAN3', 'BPSALT', 'SLEPTIM1', '_MINAC11',
       'CNCRAGE', 'EXEROFT1', '_LTASTH1', 'SDHMONEY', 'QSTLANG', 'HHADULT',
       'FLSHTMY2', 'BPALCHOL', 'ASDRVIST', 'FRUIT2', 'CRGVEXPT', 'IMFVPLAC',
       'HPVADSHT', '_CRACE1', 'CSRVINSR', '_VEGRES1', 'BPEATHBT', 'SMOKDAY2',
       'DIFFDRES', 'FTJUDA2_', 'STOPSMK2', 'ARTHDIS2', 'CNCRDIFF', 'CASTHNO2',
       'LSTCOVRG', 'SLEPDAY1', 'EYEEXAM', 'IYEAR', 'DIFFALON', 'DOCTDIAB',
       'ASERVIST', 'FIREARM4', 'PVTRESD3', 'ARTHWGT', 'ADDEPEV2', 'INSULIN',
       '_CPRACE', 'CSRVTRT2', '_PNEUMO2', 'MEDBILL1']
rf=['CHOLMED1', '_AGE65YR', '_FLSHOT6', '_PNEUMO2', 'DIABETE3', 'TOLDHI2',
       'HAVARTH3', '_DRDXAR1', 'DIABAGE2', 'ARTHDIS2', 'LMTJOIN3', '_LMTWRK1',
       '_LMTACT1', 'ARTHSOCL', '_LMTSCL1', 'DIFFWALK', 'JOINPAI1', '_RFCHOL1',
       'DIABEYE', '_RFHLTH', 'INSULIN', '_MICHD', 'DIABEDU', 'EYEEXAM',
       'FEETCHK2', 'CHKHEMO3', 'DOCTDIAB', 'BLDSUGAR', 'FEETCHK', '_HCVU651',
       'CVDCRHD4', 'CVDINFR4', 'INTERNET', 'SHINGLE2', 'BPMEDADV', 'BPEXRADV',
       'BPHI2MR', 'BPEATHBT', 'BPEXER', 'BPSALT', 'BPEATADV', 'BPSLTADV',
       'BPALCHOL', 'BPALCADV', 'CVDSTRK3', 'CHCCOPD1', 'CHCKIDNY', 'CHCOCNCR',
       'CHCSCNCR', 'ARTHEDU', 'ARTHEXER', 'ARTHWGT', 'ARTTODAY', 'DEAF',
       'DIFFDRES', 'RDUCHART', 'RLIVPAIN', 'RDUCSTRK', 'DIFFALON', 'IMFVPLAC',
       'HAREHAB1', 'CIMEMLOS', 'STREHAB1', 'CSRVTRT2', 'CNCRTYP1', 'CNCRAGE',
       'CNCRDIFF', 'CSRVSUM', 'CSRVDEIN', 'CSRVINSR', 'CSRVDOC1', 'CSRVRTRN',
       'CSRVPAIN', 'CSRVCLIN', 'CDHOUSE', 'CDASSIST', 'CDSOCIAL', 'CDDISCUS',
       'CDHELP', 'CSRVCTL1', 'VETERAN3', 'BLIND', 'ASNOSLEP', 'IMONTH',
       'PAINACT2', 'RSNMRJNA', 'QLHLTH2', 'QLSTRES2', 'QLMENTL2', 'DLYOTHER',
       '_PSU', 'SEQNO', '_LLCPWT', 'CSRVINST', 'ASYMPTOM', 'ASTHMED3',
       'COLGHOUS', 'LADULT', 'ASRCHKUP', 'ASACTLIM']
xgb = ['_BMI5', 'BPEATHBT', '_AGE80', 'WEIGHT2', 'GENHLTH', 'MAXVO2_',
       'CHOLMED1', 'IDATE', 'DIABAGE2', 'IDAY', '_STSTR', 'SEQNO', '_VEGESU1',
       '_STRWT', 'WTKG3', '_LLCPWT', '_LLCPWT2', 'PHYSHLTH', 'CHECKUP1',
       'JOINPAI1', 'FRUIT2', '_MICHD', 'POTADA1_', 'MARITAL', 'FRENCHF1',
       '_WT2RAKE', '_DRNKWEK', '_FRUTSU1', 'FVGREEN1', 'CHOLCHK1', 'INCOME2',
       'GRENDA1_', 'VEGETAB2', 'POTATOE1', 'MENTHLTH', 'VEGEDA2_', 'EDUCA',
       'PERSDOC2', 'FRNCHDA_', 'DIABETE3', 'STRENGTH', '_DUALCOR', 'POORHLTH',
       'HEIGHT3', 'HIVTSTD3', 'MAXDRNKS', 'CHCKIDNY', 'ALCDAY5', 'FLSHTMY2',
       'EMPLOY1', 'CVDASPRN', '_STATE', 'PAMIN21_', 'METVL21_', '_RFBMI5',
       '_PRACE1', 'FRUITJU2', 'PNEUVAC3', '_PHYS14D', 'FMONTH', 'PAFREQ1_',
       'FTJUDA2_', 'FRUTDA2_', 'EXERHMM1', 'PA1MIN_', 'DROCDY3_', '_MINAC11',
       '_MINAC21', 'EXRACT21', 'EXRACT11', 'PADUR2_', 'EXEROFT1', 'DIFFWALK',
       'PREGNANT', '_AGEG5YR', '_CLLCPWT', 'EXEROFT2', 'PAFREQ2_', 'STRFREQ_',
       'AVEDRNK2', '_CHISPNC', 'INTERNET', 'HTIN4', 'HHADULT', 'WTCHSALT',
       'PADUR1_', '_IMPRACE', '_RFCHOL1', 'PDIABTST', 'IMFVPLAC', 'TOLDHI2',
       'METVL11_', 'BLDSUGAR', 'EXERHMM2', 'CVDCRHD4', 'PA1VIGM_', 'DRNK3GE5',
       'SSBSUGR2', 'MARIJANA', 'DRVISITS']
tmp = [val for val in lasso if val in rf]
tmp = [val for val in tmp if val in xgb]
print(tmp)
'''
['_RFCHOL1', 'DIABAGE2', 'TOLDHI2', 'JOINPAI1', 'INTERNET', 'IMFVPLAC', 'BPEATHBT']
'''