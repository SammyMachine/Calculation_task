from math import sqrt, exp
import matplotlib.pyplot as plt
import numpy as np
from scipy import integrate
from scipy.special import erf
from scipy.stats import norm
from scipy.stats.distributions import chi2

"""1.1"""

file = open("data.txt")
data = list(map(float, file.read().splitlines()[3:]))
sorted_data = sorted(data)
vertical = np.arange(0, 1, 1 / len(sorted_data))
plt.plot(sorted_data, vertical)
plt.xlabel("X")
plt.ylabel("F(X)")
plt.show()

"""1.2"""

plt.clf()
plt.xlabel("X")
plt.ylabel("PHI(X)")
histogram = plt.hist(data, 10, density=True)
plt.show()

"""2.1.1"""

average = round(sum(data) / len(data), 6)
median = sorted_data[int(len(sorted_data) / 2 + 0.5)]
middle = (sorted_data[0] + sorted_data[-1]) / 2
print("All: First starting moment", average, ";", "Median", median, ";", "Middle span", middle)

"""2.1.2"""


def k_Moment(data, degree, average):
    result = 0.0
    for i in range(0, len(data)):
        result += (data[i] - average) ** degree
    return round(result / len(data), 6)


second_Moment = k_Moment(data, 2, average)
third_Moment = k_Moment(data, 3, average)
fourth_Moment = k_Moment(data, 4, average)
print("All: Second moment", second_Moment, ";", "Third moment", third_Moment, ";", "Fourth moment", fourth_Moment)

"""2.1.3"""

skewness = third_Moment / (second_Moment ** (3 / 2))
kurtosis = fourth_Moment / (second_Moment ** 2)
print("All: Skewness", round(skewness, 6), ";", "Kurtosis", round(kurtosis, 6))

"""2.1.4"""


def index(vertical, value):
    for i in range(0, len(vertical)):
        if vertical[i] >= value:
            return i
    return False


P = 0.95
upper_Index = (index(vertical, (1.0 + P) / 2))
bottom_Index = (index(vertical, (1.0 - P) / 2))
print("All: Upper bound quantile", sorted_data[upper_Index], ";", "Bottom bound quantile", sorted_data[bottom_Index])

"""2.2"""

data20 = data[19:39]
sorted_data20 = sorted(data20)
average20 = round(sum(data20) / len(data20), 6)
median20 = sorted_data20[int(len(sorted_data20) / 2 + 0.5)]
middle20 = (sorted_data20[0] + sorted_data20[-1]) / 2
print("20: First starting moment", average20, ";", "Median", median20, ";", "Middle span", middle20)


def k_Moment(data, degree, average):
    result = 0.0
    for i in range(0, len(data)):
        result += (data[i] - average) ** degree
    return round(result / len(data), 6)


second_Moment20 = k_Moment(data20, 2, average20)
third_Moment20 = k_Moment(data20, 3, average20)
fourth_Moment20 = k_Moment(data20, 4, average20)
print("20: Second moment", second_Moment20, ";", "Third moment", third_Moment20, ";", "Fourth moment", fourth_Moment20)


skewness20 = third_Moment20 / (second_Moment20 ** (3 / 2))
kurtosis20 = fourth_Moment20 / (second_Moment20 ** 2)
print("20: Skewness", round(skewness20, 6), ";", "Kurtosis", round(kurtosis20, 6))

"""2.3.1"""

Q = 0.95
k1 = 1.6595
k1_20 = 1.7247


def sigma(data):
    s = 0
    for i in data:
        s += (i - average) ** 2
    s /= (len(data) - 1)
    s = sqrt(s)
    return s


sigma_all = sigma(data)
sigma_20 = sigma(data20)

expectation_estimate_all_bottom = average - k1 * sigma_all / sqrt(len(data))
expectation_estimate_all_upper = average + k1 * sigma_all / sqrt(len(data))

expectation_estimate_20_bottom = average20 - k1_20 * sigma_20 / sqrt(len(data20))
expectation_estimate_20_upper = average20 + k1_20 * sigma_20 / sqrt(len(data20))

print("Expectation estimate, All:", (round(expectation_estimate_all_bottom, 6), round(expectation_estimate_all_upper, 6)))
print("Expectation estimate, 20:", (round(expectation_estimate_20_bottom, 6), round(expectation_estimate_20_upper, 6)))

k1_all = chi2.ppf((1 - Q) / 2, 105)
k2_all = chi2.ppf((1 + Q) / 2, 105)

k1__20 = chi2.ppf((1 - Q) / 2, 20)
k2__20 = chi2.ppf((1 + Q) / 2, 20)

print("k1 and k2, All:", (round(k1_all, 6), round(k2_all, 6)))
print("k1 and k2, 20:", (round(k1__20, 6), round(k2__20, 6)))

sigma_variance_all = sigma_all ** 2
sigma_variance_20 = sigma_20 ** 2

dispersion_all_bottom = sigma_variance_all * (len(data) - 1) / k2_all
dispersion_all_upper = sigma_variance_all * (len(data) - 1) / k1_all

print("Interval estimation of variance, All:", (round(dispersion_all_bottom, 6), round(dispersion_all_upper, 6)))

dispersion_20_bottom = sigma_variance_20 * (len(data) - 1) / k2__20
dispersion_20_upper = sigma_variance_20 * (len(data) - 1) / k1__20

print("Interval estimation of variance, 20:", (round(dispersion_20_bottom, 6), round(dispersion_20_upper, 6)))

"""2.3.2"""

bottom = average - sigma_all * 2.2261
upper = average + sigma_all * 2.2261

print("Parametric:", (round(bottom, 6), round(upper, 6)))
print("Nonparametric, k-1/2:", sorted_data[0], sorted_data[-2])
print("Nonparametric, k+1/2:", sorted_data[1], sorted_data[-1])

"""3"""
"""Normal distribution"""

start = histogram[1][0]
end = histogram[1][-1]
x = np.linspace(start, end, 100)
norm_distr = []
for i in x:
    v = 1 / (sqrt(2 * 3.1415 * second_Moment)) * exp(-(i-average) ** 2 / (2 * second_Moment))
    norm_distr.append(v)

plt.clf()
plt.plot(x, norm_distr)
plt.xlabel("X")
plt.ylabel("PHI(X)")
histogram = plt.hist(data, 10, density=True)
plt.show()

start = min(data)
end = max(data)
x = np.linspace(start, end, 100)
y = (1 + erf((x - average) / (sqrt(2 * second_Moment)))) / 2

plt.clf()
plt.plot(x, y)
plt.plot(sorted_data, vertical)
plt.xlabel("X")
plt.ylabel("F(X)")
plt.show()

"""Xi-criterion"""

def phi(x, m, sigma):
    return 1 / (sigma * sqrt(2 * 3.1415)) * exp(-(x - m) ** 2 / (2 * sigma ** 2))

def calc_p(start, end, sigma, m):
    return integrate.quad(lambda x: phi(x, m, sigma), start, end)[0]

Xi = 0
histogram_new = plt.hist(data, bins=10)
for i in range(0, 10):
    nk = histogram_new[0][i]
    start = histogram_new[1][i]
    end = histogram_new[1][i + 1]
    pk = calc_p(start, end, sigma_all, average)
    Xi += (nk - len(data) * pk) ** 2 / (len(data) * pk)
print("Xi-criterion:", Xi)

"""Kolmogorov-Smirnov criterion"""

y = lambda x: (1 + erf((x - average) / (sqrt(2 * second_Moment)))) / 2
value = 0.0
for i in range(len(sorted_data)):
    value = max(value, abs(vertical[i] - y(sorted_data[i])))

print("Kolmogorov-Smirnov criterion:", value)

"""Mises criterion"""

mises = 1 / (12 * len(data))
for i in range(0, len(data)):
    F = norm.cdf(sorted_data[i], average, sigma_all)
    mises += (F - (2 * i - 1) / (2 * len(data))) ** 2

print("Mises criterion:", mises)