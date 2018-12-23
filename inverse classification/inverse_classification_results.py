import  numpy as np

#2_naked:
naked_2 = np.array([[0.6984126984126984, 0.3622880332147765],
[0.5714285714285714, 0.440150868229461],
[0.8095238095238095, 0.3225553659084296],
[0.8412698412698413, 0.1726171253061654],
[0.38095238095238093, 0.5996755467597744]])

print("2_naked: ", np.around(naked_2, decimals=2))
print("2_naked_accuracy_mean_std: ", np.around(np.mean(naked_2[:,0]),2), np.around(np.std(naked_2[:,0]),2))
print("2_naked_error_mean_std: ", np.around(np.mean(naked_2[:,1]),2), np.around(np.std(naked_2[:,1]), 2))

#2_ds
ds_2 = np.array([[0.5873015873015873, 0.48370579609339315],
[0.42857142857142855, 0.5764464341684693],
[0.3968253968253968, 0.4961492109753141],
[0.746031746031746, 0.3256718465647067],
[0.5238095238095238, 0.5576492647733844]])

print("2_ds: ", np.around(ds_2, decimals=2))
print("2_ds_accuracy_mean_std: ", np.around(np.mean(ds_2[:,0]),2), np.around(np.std(ds_2[:,0]),2))
print("2_ds_error_mean_std: ", np.around(np.mean(ds_2[:,1]),2), np.around(np.std(ds_2[:,1]), 2))

#2_noise_01:
noise_2_01 = np.array([[0.9047619047619048, 0.1315291134566969],
[0.9206349206349206, 0.10450408448673962],
[0.6507936507936508, 0.41821481911922254],
[0.4603174603174603, 0.43645874488716335],
[0.4126984126984127, 0.41279800669115124]])

print("2_noise_01: ", np.around(noise_2_01, decimals=2))
print("2_noise_01_accuracy_mean_std: ", np.around(np.mean(noise_2_01[:,0]),2), np.around(np.std(noise_2_01[:,0]),2))
print("2_noise_01_error_mean_std: ", np.around(np.mean(noise_2_01[:,1]),2), np.around(np.std(noise_2_01[:,1]), 2))

#2_noise_02:
noise_2_02= np.array([[1.0, 0.0759593499013347],
[0.9682539682539683, 0.09276134912124992],
[1.0, 0.018963177939329164],
[0.9365079365079365, 0.15582147132465818],
[1.0, 0.07781698176242056]])

print("2_noise_02: ", np.around(noise_2_02, decimals=2))
print("2_noise_02_accuracy_mean_std: ", np.around(np.mean(noise_2_02[:,0]),2), np.around(np.std(noise_2_02[:,0]),2))
print("2_noise_02_error_mean_std: ", np.around(np.mean(noise_2_02[:,1]),2), np.around(np.std(noise_2_02[:,1]), 2))

#2_noise_03
noise_2_03 = np.array([[1.0, 0.04146389184518892],
[1.0, 0.09865228048290489],
[1.0, 0.12268841968810419],
[1.0, 0.06950291878573156],
[1.0, 0.09061335822971789]])

print("2_noise_03: ", np.around(noise_2_03, decimals=2))
print("2_noise_03_accuracy_mean_std: ", np.around(np.mean(noise_2_03[:,0]),2), np.around(np.std(noise_2_03[:,0]),2))
print("2_noise_03_error_mean_std: ", np.around(np.mean(noise_2_03[:,1]),2), np.around(np.std(noise_2_03[:,1]), 2))

#2_noise_04
noise_2_04 = np.array([[1.0, 0.0945932183214874],
[1.0, 0.10361514214837172],
[1.0, 0.1553425721267589],
[1.0, 0.138311098662929],
[1.0, 0.1151927357249817]])

print("2_noise_04: ", np.around(noise_2_04, decimals=2))
print("2_noise_04_accuracy_mean_std: ", np.around(np.mean(noise_2_04[:,0]),2), np.around(np.std(noise_2_04[:,0]),2))
print("2_noise_04_error_mean_std: ", np.around(np.mean(noise_2_04[:,1]),2), np.around(np.std(noise_2_04[:,1]), 2))


#2_repeller

repeller_2 = np.array([[0.8829787234042553, 0.11834418158637727],
[0.6914893617021277, 0.15382725841758407],
[0.723404255319149, 0.17655094738258284],
[0.8404255319148937, 0.15140733357219216],
[0.8297872340425532, 0.14625343689692918]])

print("2_repeller: ", np.around(repeller_2, decimals=2))
print("2_repeller_accuracy_mean_std: ", np.around(np.mean(repeller_2[:,0]),2), np.around(np.std(repeller_2[:,0]),2))
print("2_repeller_error_mean_std: ", np.around(np.mean(repeller_2[:,1]),2), np.around(np.std(repeller_2[:,1]), 2))

#2_types

types_2 = np.array([[0.5714285714285714, 0.46909085245645515],
[0.6031746031746031, 0.4460858245199946],
[0.9365079365079365, 0.15033957250802385],
[1.0, 0.2812244623341086],
[0.8412698412698413, 0.33130075138062076]])

print("2_types: ", np.around(types_2, decimals=2))
print("2_types_accuracy_mean_std: ", np.around(np.mean(types_2[:,0]),2), np.around(np.std(types_2[:,0]),2))
print("2_types_error_mean_std: ", np.around(np.mean(types_2[:,1]),2), np.around(np.std(types_2[:,1]), 2))

#2_types_noise

types_2_noise_04 = np.array([[1.0, 0.18220864906477605],
[0.9365079365079365, 0.2904404586609495],
[0.9523809523809523, 0.24518237105749247],
[0.3968253968253968, 0.543596142872431],
[0.9841269841269841, 0.32642376233545495]])

print("2_types_noise_04: ", np.around(types_2_noise_04, decimals=2))
print("2_types_accuracy_mean_std_noise_04: ", np.around(np.mean(types_2_noise_04[:,0]),2), np.around(np.std(types_2_noise_04[:,0]),2))
print("2_types_error_mean_std_noise_04: ", np.around(np.mean(types_2_noise_04[:,1]),2), np.around(np.std(types_2_noise_04[:,1]), 2))

#2_sparse4_noise

sparse_2_noise_04 = np.array([[1.0, 0.16022956536268024],
[0.9365079365079365, 0.10993446524355528],
[1.0, 0.0946226152179276],
[1.0, 0.15347101859370987],
[0.9365079365079365, 0.14455421041336294]])

print("2_sparse_noise_04: ", np.around(sparse_2_noise_04, decimals=2))
print("2_sparse_accuracy_mean_std_noise_04: ", np.around(np.mean(sparse_2_noise_04[:,0]),2), np.around(np.std(sparse_2_noise_04[:,0]),2))
print("2_sparse_error_mean_std_noise_04: ", np.around(np.mean(sparse_2_noise_04[:,1]),2), np.around(np.std(sparse_2_noise_04[:,1]), 2))

#2_sparse1_noise

sparse1_2_noise_04 = np.array([[1.0, 0.15154102144854614],
[1.0, 0.13734922995186508],
[1.0, 0.10201477310041672],
[1.0, 0.16018938842582137],
[1.0, 0.11483627603459963]])

print("2_sparse1_noise_04: ", np.around(sparse1_2_noise_04, decimals=2))
print("2_sparse1_accuracy_mean_std_noise_04: ", np.around(np.mean(sparse1_2_noise_04[:,0]),2), np.around(np.std(sparse1_2_noise_04[:,0]),2))
print("2_sparse1_error_mean_std_noise_04: ", np.around(np.mean(sparse1_2_noise_04[:,1]),2), np.around(np.std(sparse1_2_noise_04[:,1]), 2))


#4_naked
naked_4 = np.array([[0.6666666666666666, 0.29035640503089677],
[0.6178861788617886, 0.26883731865183386],
[0.4715447154471545, 0.3765823590863229],
[0.35772357723577236, 0.3974809517342973],
[0.3170731707317073, 0.39477288450459996]])


print("4_naked: ", np.around(naked_4, decimals=2))
print("4_naked_accuracy_mean_std: ", np.around(np.mean(naked_4[:,0]),2), np.around(np.std(naked_4[:,0]),2))
print("4_naked_error_mean_std: ", np.around(np.mean(naked_4[:,1]),2), np.around(np.std(naked_4[:,1]), 2))

#4_ds
ds_4 = np.array([[0.926829268292683, 0.18317829248988451],
[0.6585365853658537, 0.27891211480933614],
[0.4959349593495935, 0.3541979557380771],
[0.8699186991869918, 0.2461335305931693],
[0.3008130081300813, 0.44241515468710124]])


print("4_ds: ", np.around(ds_4, decimals=2))
print("4_ds_accuracy_mean_std: ", np.around(np.mean(ds_4[:,0]),2), np.around(np.std(ds_4[:,0]),2))
print("4_ds_error_mean_std: ", np.around(np.mean(ds_4[:,1]),2), np.around(np.std(ds_4[:,1]), 2))


#4_noise_01
noise_4_01 = np.array([[0.7642276422764228, 0.24965981520212896],
[0.943089430894309, 0.19444602880354953],
[0.8617886178861789, 0.19448341644680148],
[0.9105691056910569, 0.20621465980291317],
[0.9024390243902439, 0.23599099838380683]])

print("4_noise_01: ", np.around(noise_4_01, decimals=2))
print("4_noise_01_accuracy_mean_std: ", np.around(np.mean(noise_4_01[:,0]),2), np.around(np.std(noise_4_01[:,0]),2))
print("4_noise_01_error_mean_std: ", np.around(np.mean(noise_4_01[:,1]),2), np.around(np.std(noise_4_01[:,1]), 2))


#4_noise_02
noise_4_02 = np.array([[0.8211382113821138, 0.2585797859500031],
[0.8780487804878049, 0.20904802689437518],
[0.943089430894309, 0.20078428847966995],
[0.9349593495934959, 0.24192132403446912],
[0.9512195121951219, 0.172837467749505]])

print("4_noise_02: ", np.around(noise_4_02, decimals=2))
print("4_noise_02_accuracy_mean_std: ", np.around(np.mean(noise_4_02[:,0]),2), np.around(np.std(noise_4_02[:,0]),2))
print("4_noise_02_error_mean_std: ", np.around(np.mean(noise_4_02[:,1]),2), np.around(np.std(noise_4_02[:,1]), 2))

#4_noise_03
noise_4_03 = np.array([[0.8699186991869918, 0.20282635190369064],
[0.9186991869918699, 0.23873642571725517],
[0.959349593495935, 0.17553320665045163],
[0.9512195121951219, 0.1764110441960691],
[0.926829268292683, 0.17881755923439077]])

print("4_noise_03: ", np.around(noise_4_03, decimals=2))
print("4_noise_03_accuracy_mean_std: ", np.around(np.mean(noise_4_03[:,0]),2), np.around(np.std(noise_4_03[:,0]),2))
print("4_noise_03_error_mean_std: ", np.around(np.mean(noise_4_03[:,1]),2), np.around(np.std(noise_4_03[:,1]), 2))

#4_noise_04
noise_4_04 = np.array([[0.8943089430894309, 0.19126680009736657],
[0.983739837398374, 0.15279236800147172],
[0.9349593495934959, 0.1976159752885933],
[0.9105691056910569, 0.17611646847260332],
[0.926829268292683, 0.16672085480531426]])

print("4_noise_04: ", np.around(noise_4_04, decimals=2))
print("4_noise_04_accuracy_mean_std: ", np.around(np.mean(noise_4_04[:,0]),2), np.around(np.std(noise_4_04[:,0]),2))
print("4_noise_04_error_mean_std: ", np.around(np.mean(noise_4_04[:,1]),2), np.around(np.std(noise_4_04[:,1]), 2))


#4_repeller

repeller_4 = np.array([[0.5882352941176471, 0.29488212657790747],
[0.6470588235294118, 0.2557345978413034],
[0.6928104575163399, 0.21124730458830057],
[0.3660130718954248, 0.3615542318719822],
[0.6078431372549019, 0.24401547685085048]])

print("4_repeller: ", np.around(repeller_4, decimals=2))
print("4_repeller_accuracy_mean_std: ", np.around(np.mean(repeller_4[:,0]),2), np.around(np.std(repeller_4[:,0]),2))
print("4_repeller_error_mean_std: ", np.around(np.mean(repeller_4[:,1]),2), np.around(np.std(repeller_4[:,1]), 2))


#4_types

types_4 = np.array([[0.17886178861788618, 0.4546071904880347],
[0.24390243902439024, 0.4331471051596904],
[0.36585365853658536, 0.42840429019105547],
[0.1951219512195122, 0.4747256046622158],
[0.34146341463414637, 0.42425736186879576]])

print("4_types: ", np.around(types_4, decimals=2))
print("4_types_accuracy_mean_std: ", np.around(np.mean(types_4[:,0]),2), np.around(np.std(types_4[:,0]),2))
print("4_types_error_mean_std: ", np.around(np.mean(types_4[:,1]),2), np.around(np.std(types_4[:,1]), 2))

#4_types_noise

types_4_noise_04 = np.array([[0.9024390243902439, 0.3696188513652769],
[0.8780487804878049, 0.352262248703735],
[0.8617886178861789, 0.3115693293780409],
[0.7154471544715447, 0.35710170485962317],
[0.6097560975609756, 0.37107162652515296]])

print("4_types_noise_04: ", np.around(types_4_noise_04, decimals=2))
print("4_types_noise_04_accuracy_mean_std: ", np.around(np.mean(types_4_noise_04[:,0]),2), np.around(np.std(types_4_noise_04[:,0]),2))
print("4_types_noise_04_error_mean_std: ", np.around(np.mean(types_4_noise_04[:,1]),2), np.around(np.std(types_4_noise_04[:,1]), 2))

#4_sparse_noise

sparse_4_noise_04 = np.array([[0.983739837398374, 0.16932136289016878],
[0.959349593495935, 0.20415932119235822],
[0.9349593495934959, 0.17441658694368803],
[0.9512195121951219, 0.1742195820294013],
[1.0, 0.17104229645150246]])

print("4_sparse_noise_04: ", np.around(sparse_4_noise_04, decimals=2))
print("4_sparse_noise_04_accuracy_mean_std: ", np.around(np.mean(sparse_4_noise_04[:,0]),2), np.around(np.std(sparse_4_noise_04[:,0]),2))
print("4_sparse_noise_04_error_mean_std: ", np.around(np.mean(sparse_4_noise_04[:,1]),2), np.around(np.std(sparse_4_noise_04[:,1]), 2))

sparse1_4_noise_04 = np.array([[0.991869918699187, 0.17959910430318454],
[0.983739837398374, 0.14908433472305846],
[0.991869918699187, 0.1932473448635198],
[0.943089430894309, 0.1887658640165149],
[0.959349593495935, 0.18454545779978818]])

print("4_sparse1_noise_04: ", np.around(sparse1_4_noise_04, decimals=2))
print("4_sparse1_noise_04_accuracy_mean_std: ", np.around(np.mean(sparse1_4_noise_04[:,0]),2), np.around(np.std(sparse1_4_noise_04[:,0]),2))
print("4_sparse1_noise_04_error_mean_std: ", np.around(np.mean(sparse1_4_noise_04[:,1]),2), np.around(np.std(sparse1_4_noise_04[:,1]), 2))

#20_naked

naked_20 = np.array([[0.39335664335664333, 0.19190033711777332],
[0.40734265734265734, 0.19686802699245137],
[0.34440559440559443, 0.19921504708468532],
[0.3409090909090909, 0.198307514005876969],
[0.2517482517482518, 0.20210949600838998]])

print("20_naked: ", np.around(naked_20, decimals=2))
print("20_naked_accuracy_mean_std: ", np.around(np.mean(naked_20[:,0]),2), np.around(np.std(naked_20[:,0]),2))
print("20_naked_error_mean_std: ", np.around(np.mean(naked_20[:,1]),2), np.around(np.std(naked_20[:,1]), 2))

#20_ds

ds_20 = np.array([[0.21503496503496503, 0.21075054712178956],
[0.288, 0.21200178565328368],
[0.5, 0.19402739184876183],
[0.308, 0.20256596560436818],
[0.308, 0.1987081242817012]])

print("20_ds: ", np.around(ds_20, decimals=2))
print("20_ds_accuracy_mean_std: ", np.around(np.mean(ds_20[:,0]),2), np.around(np.std(ds_20[:,0]),2))
print("20_ds_error_mean_std: ", np.around(np.mean(ds_20[:,1]),2), np.around(np.std(ds_20[:,1]), 2))


#20_repeller

repeller_20 = np.array([[0.21166666666666667, 0.20285098156036369],
[0.344, 0.19663184491854235],
[0.26, 0.201076711861049],
[0.328, 0.19405996022873712],
[0.328, 0.19646663964660338]])

print("20_repeller: ", np.around(repeller_20, decimals=2))
print("20_repeller_accuracy_mean_std: ", np.around(np.mean(repeller_20[:,0]),2), np.around(np.std(repeller_20[:,0]),2))
print("20_repeller_error_mean_std: ", np.around(np.mean(repeller_20[:,1]),2), np.around(np.std(repeller_20[:,1]), 2))

#20_noise_01

noise_20_01 = np.array([[0.36713286713286714, 0.20149228099303645],
[0.34, 0.19910752092504524],
[0.372, 0.1963380911374035],
[0.42, 0.1857226967616749],
[0.236, 0.21035112632220035]])

print("20_noise_01: ", np.around(noise_20_01, decimals=2))
print("20_noise_01_accuracy_mean_std: ", np.around(np.mean(noise_20_01[:,0]),2), np.around(np.std(noise_20_01[:,0]),2))
print("20_noise_01_error_mean_std: ", np.around(np.mean(noise_20_01[:,1]),2), np.around(np.std(noise_20_01[:,1]), 2))

#20_noise_02

noise_20_02 = np.array([[0.5052447552447552, 0.19400801507407445],
[0.224, 0.19871079805019518],
[0.448, 0.1918744037711736],
[0.268, 0.20039397481095558],
[0.444, 0.18281824333798652]])

print("20_noise_02: ", np.around(noise_20_02, decimals=2))
print("20_noise_02_accuracy_mean_std: ", np.around(np.mean(noise_20_02[:,0]),2), np.around(np.std(noise_20_02[:,0]),2))
print("20_noise_02_error_mean_std: ", np.around(np.mean(noise_20_02[:,1]),2), np.around(np.std(noise_20_02[:,1]), 2))

#20_noise_03

noise_20_03 = np.array([[0.6468531468531469, 0.16894969981297742],
[0.42, 0.17911248174116998],
[0.332, 0.19313180924807596],
[0.328, 0.19260185932648646],
[0.412, 0.18209938833869632]])

print("20_noise_03: ", np.around(noise_20_03, decimals=2))
print("20_noise_03_accuracy_mean_std: ", np.around(np.mean(noise_20_03[:,0]),2), np.around(np.std(noise_20_03[:,0]),2))
print("20_noise_03_error_mean_std: ", np.around(np.mean(noise_20_03[:,1]),2), np.around(np.std(noise_20_03[:,1]), 2))

#20_noise_04

noise_20_04 = np.array([[0.3321678321678322, 0.1901968072103566],
[0.152, 0.2053412442389766],
[0.116, 0.2052003312197174],
[0.236, 0.1961869036087434],
[0.164, 0.2030803514917294]])

print("20_noise_04: ", np.around(noise_20_04, decimals=2))
print("20_noise_04_accuracy_mean_std: ", np.around(np.mean(noise_20_04[:,0]),2), np.around(np.std(noise_20_04[:,0]),2))
print("20_noise_04_error_mean_std: ", np.around(np.mean(noise_20_04[:,1]),2), np.around(np.std(noise_20_04[:,1]), 2))

#20_types

types_20 = np.array([[0.068, 0.21604259766633585],
[0.076, 0.2183448431531461],
[0.14, 0.2177983567571351],
[0.048, 0.21752558243470982],
[0.088, 0.2167072762343738]])

print("20_types: ", np.around(types_20, decimals=2))
print("20_types_accuracy_mean_std: ", np.around(np.mean(types_20[:,0]),2), np.around(np.std(types_20[:,0]),2))
print("20_types_error_mean_std: ", np.around(np.mean(types_20[:,1]),2), np.around(np.std(types_20[:,1]), 2))

#20_types_noise_03

types_20_noise_03 = np.array([[0.17482517482517482, 0.21452139423881966],
[0.056, 0.22030373280394372],
[0.076, 0.2192249469760734],
[0.06, 0.21873711599295692],
[0.088, 0.22160960611658334]])

print("20_types_noise_03: ", np.around(types_20_noise_03, decimals=2))
print("20_types_noise_03_accuracy_mean_std: ", np.around(np.mean(types_20_noise_03[:,0]),2), np.around(np.std(types_20_noise_03[:,0]),2))
print("20_types_noise_03_error_mean_std: ", np.around(np.mean(types_20_noise_03[:,1]),2), np.around(np.std(types_20_noise_03[:,1]), 2))

#20_sparse_noise_03

sparse_20_noise_03 = np.array([[0.4737762237762238, 0.18141174699734308],
[0.3933333333333333, 0.197688751882428],
[0.488, 0.17902744548971156],
[0.5, 0.1859198257257685],
[0.58, 0.18218082985919257]])

print("20_sparse_noise_03: ", np.around(sparse_20_noise_03, decimals=2))
print("20_sparse_noise_03_accuracy_mean_std: ", np.around(np.mean(sparse_20_noise_03[:,0]),2), np.around(np.std(sparse_20_noise_03[:,0]),2))
print("20_sparse_noise_03_error_mean_std: ", np.around(np.mean(sparse_20_noise_03[:,1]),2), np.around(np.std(sparse_20_noise_03[:,1]), 2))


sparse1_20_noise_03 = np.array([[0.588, 0.17694291826108127],
[0.584, 0.18038509533006683],
[0.628, 0.17384842811740792],
[0.652, 0.17134160798328515],
[0.636, 0.1733644215401092]])

print("20_sparse1_noise_03: ", np.around(sparse1_20_noise_03, decimals=2))
print("20_sparse1_noise_03_accuracy_mean_std: ", np.around(np.mean(sparse1_20_noise_03[:,0]),2), np.around(np.std(sparse1_20_noise_03[:,0]),2))
print("20_sparse1_noise_03_error_mean_std: ", np.around(np.mean(sparse1_20_noise_03[:,1]),2), np.around(np.std(sparse1_20_noise_03[:,1]), 2))




"""""""""""""""Comparison"""""""""""""""

#2_comparison

comparison_2_mse = np.array([[1.0, 1.2236199964625849e-05],
[1.0, 1.0650903584561553e-05],
[1.0, 1.1003356833632673e-05],
[1.0, 5.332670389234608e-05],
[1.0, 8.325256171557485e-06]])

print("2_comparison_mse: ", np.around(comparison_2_mse, decimals=2))
print("2_comparison_mse_accuracy_mean_std: ", np.around(np.mean(comparison_2_mse[:,0]),2), np.around(np.std(comparison_2_mse[:,0]),2))
print("2_comparison_mse_error_mean_std: ", np.around(np.mean(comparison_2_mse[:,1]),2), np.around(np.std(comparison_2_mse[:,1]), 2))

#2_comparison_types

comparison_2_types = np.array([[1.0, 6.614138431932695e-05],
[0.9682539682539683, 0.17813370201272277],
[0.9682539682539683, 0.17814000948079323],
[0.9523809523809523, 0.21719142246512],
[1.0, 1.6050436562876023e-05]])

print("2_comparison_sparse: ", np.around(comparison_2_types, decimals=2))
print("2_comparison_sparse_accuracy_mean_std: ", np.around(np.mean(comparison_2_types[:,0]),2), np.around(np.std(comparison_2_types[:,0]),2))
print("2_comparison_sparse_error_mean_std: ", np.around(np.mean(comparison_2_types[:,1]),2), np.around(np.std(comparison_2_types[:,1]), 2))

#4_comparison

comparison_4_mse = np.array([[0.926829268292683, 0.16363407954601153],
[0.9512195121951219, 0.11654141255359944],
[0.9186991869918699, 0.16229324072125728],
[0.9512195121951219, 0.11891059560323682],
[0.959349593495935,  0.10131213230125374]])

print("4_comparison_mse: ", np.around(comparison_4_mse, decimals=2))
print("4_comparison_mse_accuracy_mean_std: ", np.around(np.mean(comparison_4_mse[:,0]),2), np.around(np.std(comparison_4_mse[:,0]),2))
print("4_comparison_mse_error_mean_std: ", np.around(np.mean(comparison_4_mse[:,1]),2), np.around(np.std(comparison_4_mse[:,1]), 2))

#4_comparison_types

comparison_4_types = np.array([[0.9512195121951219, 0.15615045768660551],
[0.9105691056910569, 0.21139966229499868],
[0.9349593495934959, 0.1798726695904628],
[0.943089430894309, 0.16868385811922285],
[0.967479674796748, 0.1275135393159052]])

print("4_comparison_sparse: ", np.around(comparison_4_types, decimals=2))
print("4_comparison_sparse_accuracy_mean_std: ", np.around(np.mean(comparison_4_types[:,0]),2), np.around(np.std(comparison_4_types[:,0]),2))
print("4_comparison_sparse_error_mean_std: ", np.around(np.mean(comparison_4_types[:,1]),2), np.around(np.std(comparison_4_types[:,1]), 2))

#20_comparison_mse

comparison_20_mse = np.array([[0.9755244755244755, 0.0430507492091986],
[0.9685314685314685,0.04491051352399841],
[0.9527972027972028, 0.058167929996317515],
[0.9685314685314685, 0.046766395941255326],
[0.9615384615384616, 0.05022681131202582]])

print("20_comparison_mse: ", np.around(comparison_20_mse, decimals=2))
print("20_comparison_mse_accuracy_mean_std: ", np.around(np.mean(comparison_20_mse[:,0]),2), np.around(np.std(comparison_20_mse[:,0]),2))
print("20_comparison_mse_error_mean_std: ", np.around(np.mean(comparison_20_mse[:,1]),2), np.around(np.std(comparison_20_mse[:,1]), 2))

#20_comparison_types

comparison_20_types = np.array([[0.7552447552447552, 0.14831457799864442],
[0.6346153846153846, 0.1645157125879259],
[0.6031468531468531, 0.16566127975751535],
[0.3793706293706294, 0.19430169202747888],
[0.6835664335664335, 0.1546053799339494]])

print("20_comparison_sparse: ", np.around(comparison_20_types, decimals=2))
print("20_comparison_sparse_accuracy_mean_std: ", np.around(np.mean(comparison_20_types[:,0]),2), np.around(np.std(comparison_20_types[:,0]),2))
print("20_comparison_sparse_error_mean_std: ", np.around(np.mean(comparison_20_types[:,1]),2), np.around(np.std(comparison_20_types[:,1]), 2))