import os


def read_power(file):
    power,phase = [], []
    # <number> <X(m)> <Y(m)> <Z(m)> <Distance(m)> <Power(dBm)> <Phase(deg)>
    with open(file) as f:
        next_line = f.readline()
        max_p, min_p = float("-inf"), float("inf")
        while next_line:
            all = next_line.split(' ')
            pow = float(all[5:6][0])
            pha = float(all[6:][0])
            if pha != 0:
                max_p = max(pow, max_p)
                min_p = min(pow, min_p)
            power.append(pow)
            phase.append(pha)
            next_line = f.readline()
    f.close()
    return power, max_p, min_p


def read_data(file_wifi, file_5g):
    power_wifi, max_wifi, min_wifi = read_power("tune/" + file_wifi)
    power_5g, max_5g, min_5g = read_power("tune/"+ file_5g)
    print("Read power map value")
    print("    (wifi)  " + '%.4f'%min_wifi + "  " + '%.4f'%max_wifi)
    print("    (5g)  " + '%.4f'%min_5g + "  " + '%.4f'%max_5g)
    return power_wifi, power_5g


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

    percent_wifi, percent_5g, percent_any = num_wifi_available/num_points, num_5g_available/num_points, num_any_available/num_points
    print("    (wifi coverage)" + '%f'%percent_wifi, "(5g coverage)" + '%f'%percent_5g, "(overall coverage)" + '%f'%percent_any)
    return percent_wifi, percent_5g, percent_any

def parse_and_sort(root_dir):
    files_wifi, files_5g = [], []
    for f_name in os.listdir(root_dir):
        if (f_name.endswith('wifi.p2m')):
            files_wifi.append(f_name)
        if (f_name.endswith('5g.p2m')):
            files_5g.append(f_name)
    files_wifi.sort(), files_5g.sort()
    return files_wifi, files_5g


def main():
    root_dir = r"old data/tune"
    files_wifi, files_5g = parse_and_sort(root_dir)
    # print(files_wifi, files_5g)
    for file_wifi, file_5g in zip(files_wifi, files_5g):
        print("Open files: " + file_wifi + "  " + file_5g)

        power_wifi, power_5g = read_data(file_wifi, file_5g)

        threshold_wifi = -93
        threshold_5g = -76.9
        print("Set indoor/outdoor threshold: ", '%.1f'%threshold_wifi, '%.1f'%threshold_5g)

        cal_percent(power_wifi, power_5g, threshold_wifi, threshold_5g)
        print("--------------------------------------------------------------------------------")

if __name__ == '__main__':
    main()
