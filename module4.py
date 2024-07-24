import numpy as np
import matplotlib.plot as plt
from arch import arch_model


VaR = mean_forecast.values + np.sqrt(variance_forecast).values*quantile

# Specifify and fit a GARCH model
basic_gm = arch_model(bitcoin_data['Return'], p = 1, q = 1,
	mean = 'constant', vol = 'GARCH', dist = 't')
gn_result = basic_gm.fit()

# Make variance forecast
gm_forecast = gm_result.forecast(start = '2019-01-01')

mean_forecast = gm_forecast.mean['2019-01-01':]
variance_forecast = gm_forecast.variance['2019-01-01':]

# Assume a Student's t-dsitrubution
# ppf(): Percent point function

q_parametric = garch_model.distribution.ppf(0,05, nu)

q_empirical = std_resid.quantile(0.05)

# Obtain the parametric quantile
q_parametric = basic_gm.distribution.ppf(0.05, nu)
print('5% parametric quantile: ', q_parametric)
    
# Calculate the VaR
VaR_parametric = mean_forecast.values + np.sqrt(variance_forecast).values *q_parametric
# Save VaR in a DataFrame
VaR_parametric = pd.DataFrame(VaR_parametric, columns = ['5%'], index = variance_forecast.index)

# Plot the VaR
plt.plot(VaR_parametric, color = 'red', label = '5% Parametric VaR')
plt.scatter(variance_forecast.index,bitcoin_data.Return['2019-1-1':], color = 'orange', label = 'Bitcoin Daily Returns' )
plt.legend(loc = 'upper right')
plt.show()

# Obtain the empirical quantile
q_empirical = std_resid.quantile(0.05)
print('5% empirical quantile: ', q_empirical)

# Calculate the VaR
VaR_empirical = mean_forecast.values + np.sqrt(variance_forecast).values*q_empirical
# Save VaR in a DataFrame
VaR_empirical = pd.DataFrame(VaR_empirical, columns = ['5%'], index = variance_forecast.index)

# Plot the VaRs
plt.plot(VaR_empirical, color = 'brown', label = '5% Empirical VaR')
plt.plot(VaR_parametric, color = 'red', label = '5% Parametric VaR')
plt.scatter(variance_forecast.index,bitcoin_data.Return['2019-1-1':], color = 'orange', label = 'Bitcoin Daily Returns' )
plt.legend(loc = 'upper right')
plt.show()

covariance = correlation*garch_vol1*garch_vol2

# gm_eur, gm_cad are fitted GARCH models
vol_eur = gm_eur.conditional_volatility
vol_cad = gm_cad.conditional_volatility

corr = np.corrcoef(resid_eur, resid_cad)[0,1]
covariance = corr*vol_eur*vol_cad

# Calculate correlation
corr = np.corrcoef(resid_eur, resid_cad)[0,1]
print('Correlation: ', corr)

# Calculate GARCH covariance
covariance =  corr *vol_eur *vol_cad 

# Plot the data
plt.plot(covariance, color = 'gold')
plt.title('GARCH Covariance')
plt.show()

# Define weights
Wa1 = 0.9
Wa2 = 1 - Wa1
Wb1 = 0.5
Wb2 = 1 - Wb1

# Calculate portfolio variance
portvar_a = Wa1**2 * variance_eur + Wa2**2 * variance_cad + 2*Wa1*Wa2 *covariance
portvar_b = Wb1**2 * variance_eur + Wb2**2 * variance_cad + 2*Wb1*Wb2*covariance

# Plot the data
plt.plot(portvar_a, color = 'green', label = 'Portfolio a')
plt.plot(portvar_b, color = 'deepskyblue', label = 'Portfolio b')
plt.legend(loc = 'upper right')
plt.show()

resid_stock = stock_gm.resid/stock_gm.conditional_volatility
resid_sp500 = sp500_gm.resid/sp500_gm.conditional_volatility

correlation = numpy.corrcoef(resid_stock, resid_sp500)[0,1]
stock_beta = correlation*(stock_gm.conditional_volatility/sp500_gm.conditional_volatility)

# Compute correlation between SP500 and Tesla
correlation = np.corrcoef(teslaGarch_resid, spGarch_resid)[0, 1]

# Compute the Beta for Tesla
stock_beta = correlation*(teslaGarch_vol / spGarch_vol)

# Plot the Beta
plt.title('Tesla Stock Beta')
plt.plot(stock_beta)
plt.show()