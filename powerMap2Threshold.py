import os
from tabulate import tabulate
import numpy as np
import pandas as pd


def parse_and_sort(root_dir):
    files_wifi, files_5g = [], []
    for f_name in os.listdir(root_dir):
        if (f_name.endswith('wifi.p2m')):
            files_wifi.append(f_name)
        if (f_name.endswith('5g.p2m')):
            files_5g.append(f_name)
    files_wifi.sort(), files_5g.sort()

    if len(files_wifi) != len(files_5g):
        print(f"files dimension is different!!! files_wifi({len(files_wifi)}) files_5g({len(files_5g)})")
    return files_wifi, files_5g


def read_power(file):
    power,phase = [], []
    # <number> <X(m)> <Y(m)> <Z(m)> <Distance(m)> <Power(dBm)> <Phase(deg)>
    with open(file) as f:
        for _ in range(3):  # Skip the first three lines
            next(f)
        max_p, min_p = float("-inf"), float("inf")
        for next_line in f:
            all = next_line.split(' ')
            pow = float(all[5:6][0])
            pha = float(all[6:][0])
            if pha != 0:
                max_p = max(pow, max_p)
                min_p = min(pow, min_p)
            power.append(pow)
            phase.append(pha)
    f.close()
    return power, max_p, min_p


def read_data(root_dir, file_wifi, file_5g):
    material_wifi = file_wifi.split("_")[:2]
    material_5g = file_5g.split("_")[:2]
    if material_wifi != material_5g:
        print(material_wifi, material_5g)

    power_wifi, max_wifi, min_wifi = read_power(f"{root_dir}/{file_wifi}")
    power_5g, max_5g, min_5g = read_power(f"{root_dir}/{file_5g}")
    return power_wifi, max_wifi, min_wifi, power_5g, max_5g, min_5g


def cal_percent(power_wifi, power_5g, threshold_wifi, threshold_5g):
    if(len(power_wifi) != len(power_5g)):
        print("ERROR: data dimension anomaly")

    num_points = len(power_wifi)
    num_wifi_available, num_5g_available, num_any_available= 0, 0, 0
    for i in range(num_points):
        if(power_wifi[i] >= threshold_wifi):
            num_wifi_available += 1
        if(power_5g[i] >= threshold_5g):
            num_5g_available += 1
        if(power_wifi[i] >= threshold_wifi or power_5g[i] >= threshold_5g):
            num_any_available += 1

    percent_wifi, percent_5g, percent_any = round(num_wifi_available / num_points, 6), round(
        num_5g_available / num_points, 6), round(num_any_available / num_points, 6)
    # percent_wifi, percent_5g, percent_any = num_wifi_available/num_points, num_5g_available/num_points, num_any_available/num_points
    # print("    (wifi coverage)" + '%f'%percent_wifi, "(5g coverage)" + '%f'%percent_5g, "(overall coverage)" + '%f'%percent_any)
    return percent_wifi, percent_5g, percent_any


def display_result(root_dir, threshold_wifi, threshold_5g):
    files_wifi, files_5g = parse_and_sort(root_dir)

    best_line= []
    best_percent = float("-inf")
    result_table = []
    result_table.append(["Material Type (Con./Per.)", "Range", "Cov.", "Total Cov."])
    for file_wifi, file_5g in zip(files_wifi, files_5g):
        parts = file_wifi.split("_")
        con, per = parts[:2]
        mat_type = f"{con}_{per}"

        power_wifi, max_wifi, min_wifi, power_5g, max_5g, min_5g = read_data(root_dir, file_wifi, file_5g)

        percent_wifi, percent_5g, percent_any = cal_percent(power_wifi, power_5g, threshold_wifi, threshold_5g)
        if percent_any > best_percent:
            best_percent = percent_any
            best_line =[[mat_type, f"(wifi) {min_wifi} ~ {max_wifi} dBm", percent_wifi, percent_any],
                        [None, f"(5g) {min_5g} ~ {max_5g} dBm", percent_5g, None]]

        result_table.append([mat_type, f"(wifi) {min_wifi} ~ {max_wifi} dBm", percent_wifi, percent_any])
        result_table.append([None, f"(5g) {min_5g} ~ {max_5g} dBm", percent_5g, None])

    print(f"Setting threshold_wifi = {threshold_wifi}, threshold_5g = {threshold_5g}")
    table = tabulate(result_table, headers="firstrow", tablefmt="pipe")
    # pick the best one who with highest total cov.
    best = tabulate(best_line, headers=["Best Result", "", "", ""], tablefmt="pipe")
    print(table, "\n\n")
    print(best)


def getRes(root_dir, threshold_wifi, threshold_5g):
    files_wifi, files_5g = parse_and_sort(root_dir)
    con_per_wifi_5g_total = []
    best_line = []
    best_percent = float("-inf")
    for file_wifi, file_5g in zip(files_wifi, files_5g):
        parts = file_wifi.split("_")
        con, per = parts[:2]
        mat_type = f"{con}_{per}"
        power_wifi, max_wifi, min_wifi, power_5g, max_5g, min_5g = read_data(root_dir, file_wifi, file_5g)
        percent_wifi, percent_5g, percent_any = cal_percent(power_wifi, power_5g, threshold_wifi, threshold_5g)
        if percent_any > best_percent:
            best_percent = percent_any
            best_line = [[mat_type, f"(wifi) {min_wifi} ~ {max_wifi} dBm", percent_wifi, percent_any],
                         [None, f"(5g) {min_5g} ~ {max_5g} dBm", percent_5g, None]]

        # save the conductivity, permitticity and three percentage in a mat
        con_per_wifi_5g_total.append([con, per, percent_wifi, percent_5g, percent_any])

    # pick the best one who with highest total cov.
    best = tabulate(best_line, headers=["Best Result", "", "", ""], tablefmt="pipe")
    print(best)
    return con_per_wifi_5g_total


def ifPercentRepeat(root_dir, threshold_wifi, threshold_5g):
    con_per_wifi_5g_total = getRes(root_dir, threshold_wifi, threshold_5g)
    # trans to DataFrame
    data = np.array(con_per_wifi_5g_total)
    column_names = ['con', 'per', 'wifiP', '5gP', 'totalP']
    material_percent = pd.DataFrame(data, columns=column_names)
    # save 3 digits and trans to percentage
    material_percent[['wifiP', '5gP', 'totalP']] = material_percent[['wifiP', '5gP', 'totalP']].apply(pd.to_numeric, errors='coerce')
    material_percent[['wifiP', '5gP', 'totalP']] = material_percent[['wifiP', '5gP', 'totalP']].applymap(lambda x: round(x * 100, 1) if isinstance(x, (int, float)) else x)
    # print(material_percent)
    # check if duplicate exist
    duplicates = material_percent.duplicated(subset=['wifiP', '5gP', 'totalP'])
    if duplicates.any():
        print("Duplication exist：")
        pd.set_option('display.max_rows', None)
        print(material_percent[duplicates])
    else:
        print("Duplication not exist")


def toCSV(threshold_wifi, threshold_5g):
    le3 = np.array(getRes(r"e-3_n3000", threshold_wifi, threshold_5g))
    le4 = np.array(getRes(r"e-4_n580", threshold_wifi, threshold_5g))
    le5 = np.array(getRes(r"e-5_n540", threshold_wifi, threshold_5g))
    le6 = np.array(getRes(r"e-6_n540", threshold_wifi, threshold_5g))
    le7 = np.array(getRes(r"e-7_n540", threshold_wifi, threshold_5g))
    le8 = np.array(getRes(r"e-8_n540", threshold_wifi, threshold_5g))
    le9 = np.array(getRes(r"e-9_n540", threshold_wifi, threshold_5g))
    le10 = np.array(getRes(r"e-10_n540", threshold_wifi, threshold_5g))
    le11 = np.array(getRes(r"e-11_n540", threshold_wifi, threshold_5g))
    le12 = np.array(getRes(r"e-12_n540", threshold_wifi, threshold_5g))
    le13 = np.array(getRes(r"e-13_n500", threshold_wifi, threshold_5g))

    stacked_arrays = np.vstack((le3, le4, le5, le6, le7, le8, le9, le10, le11, le12, le13))
    df = pd.DataFrame(stacked_arrays)
    df.to_csv('all_data.csv', index=False, header=['con', 'per', 'wifi', '5g', 'any'])


def concatToCSV(threshold_wifi, threshold_5g):
    # l5 = np.array(getRes(r"0.005_n200", threshold_wifi, threshold_5g))
    # l9 = np.array(getRes(r"0.009_n200", threshold_wifi, threshold_5g))
    # l_lessthan_1 = np.array(getRes(r"per_lessthan_1", threshold_wifi, threshold_5g))
    # l6 = np.array(getRes(r"0.006_n200", threshold_wifi, threshold_5g))
    # l7 = np.array(getRes(r"0.007_n200", threshold_wifi, threshold_5g))
    # l8 = np.array(getRes(r"0.008_n200", threshold_wifi, threshold_5g))
    # le4 = np.array(getRes(r"e-4_n180", threshold_wifi, threshold_5g))
    # le5 = np.array(getRes(r"e-5_n540", threshold_wifi, threshold_5g))
    # le6 = np.array(getRes(r"e-6_n540", threshold_wifi, threshold_5g))
    # le7 = np.array(getRes(r"e-7_n540", threshold_wifi, threshold_5g))
    # le9 = np.array(getRes(r"e-9_n540", threshold_wifi, threshold_5g))
    # le10 = np.array(getRes(r"e-10_n540", threshold_wifi, threshold_5g))
    # le11 = np.array(getRes(r"e-11_n540", threshold_wifi, threshold_5g))
    # le12 = np.array(getRes(r"e-12_n540", threshold_wifi, threshold_5g))
    # le13 = np.array(getRes(r"e-13_n500", threshold_wifi, threshold_5g))
    le14 = np.array(getRes(r"e-14_n540", threshold_wifi, threshold_5g))
    le15 = np.array(getRes(r"e-15_n540", threshold_wifi, threshold_5g))

    stacked_arrays = np.vstack((le14, le15))
    df = pd.DataFrame(stacked_arrays)
    df.to_csv('all_data.csv', mode='a', header=False, index=False)


def repeatingPercent(filename):
    # 读取CSV文件
    df = pd.read_csv(filename)
    # 取最后三列
    last_three_columns = df.iloc[:, -3:]
    # 检查这三列是否存在重复的行
    duplicates = last_three_columns.duplicated()
    # 计算重复行的数量
    num_duplicates = duplicates.sum()
    # 计算重复的概率
    duplicate_probability = num_duplicates / len(df)
    print("the probability of repeating (last three cols)：", duplicate_probability)


def main():
    root_dir = r"e-7_n540"
    threshold_wifi = -92
    threshold_5g = -77

    # toCSV(threshold_wifi, threshold_5g)
    # display_result(root_dir, threshold_wifi, threshold_5g)
    # ifPercentRepeat(root_dir, threshold_wifi, threshold_5g)
    concatToCSV(threshold_wifi, threshold_5g)
    # getRes(root_dir, threshold_wifi, threshold_5g)

    # repeatingPercent('all_data.csv')


if __name__ == '__main__':
    main()


