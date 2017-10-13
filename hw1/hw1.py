import sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
np.set_printoptions(suppress=True)

def main(argv):
    data_table = pd.read_csv(argv[0], header = -1,encoding='Big5')
    data_hour = []
    cnt = 0
    for j in range(0,len(data_table),18):
        for i in range(2,11,1):
            data_hour.append([])
            for k in range(0,18,1):
                if 'NR' == data_table.loc[k+j][i]:
                    data_hour[cnt].append(0.0)
                else:
                    data_hour[cnt].append(float(data_table.loc[k+j][i]))
            cnt += 1
    
    x_list = []
    feature_select = [0,7,8,9,11]
    feature_select2 = [8,9]
    previous_hour = 9
    # feature_select = [9]
    # feature_select2 = []
    it = 0
    for i in range(0,cnt,9):
        x_list.append([])
        for j in range(9-previous_hour,9,1):
            for k in range(0,18,1):
                if k in feature_select:
                    x_list[it].append(data_hour[i+j][k])
                if k in feature_select2:
                    x_list[it].append(data_hour[i+j][k]**2)
        x_list[it].append(1) # bias
        it += 1

    test_data_len = it
    x_item = np.array(x_list)

    df_table = []
    w_list = [-0.03748516420160857, -0.003401649931221073, 0.0002668800425528564, 3.5018621892032245e-05, 0.023938986799668948, -0.0009086259667646033, 0.04432154609399053, -0.026553329552212533, 0.01830522448829653, -0.0015813586486583338, -2.7070990419114065e-05, -0.0017347966053462666, 0.00019738949697713673, 0.005104883382680513, -0.015459163152264069, 0.004664055856711743, -0.003544787811806847, 6.379798256753393e-05, 0.020919681482356196, 0.0016893951590237842, -0.002397861750004949, -0.01706300643305495, -0.004528964730145138, 0.017561539235128803, 6.658989797156445e-05, -0.029807879254417653, -0.0020767052675565983, -0.017540263016597547, -0.006231901579900382, -0.007428679308211496, 0.023522005349625143, -0.0005607028464281242, 0.027817703499800252, -9.886400961624491e-05, -0.010340490086193877, -0.0012374579039504136, -0.01154702371692983, -0.00731761459870927, 0.00015272236803383504, 0.08558013873615891, 0.004210551835874319, 0.021796938708357833, 0.0042813555200851925, -0.009457288271123897, -0.016738423540693374, 0.0003066300292471308, -0.1497123253348895, -0.005004718235002591, -0.02046650415822171, 0.013242572175545005, -0.017723346254223796, 0.019847284216974844, -0.0005250314223626131, 0.14645032215344284, -0.0012125832950948766, -0.007451932866869423, 0.045212064797839284, 0.05308996723092196, 0.08167030878475874, 0.00014039372708101152, 0.6531523871910044, 0.003956633489226698, -0.007019737846992183, 0.00277956154722]
    w = np.array(w_list)
 
    for i in range(0,test_data_len,1):
        df_table.append(['id_' + str(i), np.dot(w,x_item[i])])

    df = pd.DataFrame.from_records(df_table, columns = ["id","value"])
    df.to_csv(argv[1],index=False)

if __name__ == '__main__':
    main(sys.argv[1:])