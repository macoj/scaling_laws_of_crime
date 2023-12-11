from scaling import analysis
import pandas as pd
import numpy as np

pure_model = ['gaussian', 'gaussian_delta_fixed', 'lognormal', 'lognormal_delta_fixed']

models = {'lognormal_delta_fixed': analysis.LogNormalFixedDAnalysis,
          'lognormal_delta_fixed_beta_fixed': analysis.LogNormalFixedDFixedBetaAnalysis,
          'lognormal': analysis.LogNormalAnalysis,
          'lognormal_beta_fixed': analysis.LogNormalFixedBetaAnalysis,
          'gaussian_delta_fixed': analysis.FixedDAnalysis,
          'gaussian_delta_fixed_beta_fixed': analysis.FixedDFixedBetaAnalysis,
          'gaussian': analysis.ConstrainedDAnalysis,
          'gaussian_beta_fixed': analysis.ConstrainedDFixedBetaAnalysis,
          'person': analysis.PopulationAnalysis,
          'person_beta_fixed': analysis.PopulationFixedGammaAnalysis}


def fit_models(data_xy, required_successes=1):
    errors = {}
    means = {}
    stds = {}
    results = []
    for m in models:
        model = models[m]
        r = model(data_xy, required_successes=required_successes)
        beta, beta_ci = r.beta
        bic = r.bic
        params = list(r.params)
        for _ in range(4 - len(params)):
            params.append("")
        result = [m, beta, beta_ci, bic, r.p_value]
        for p in params:
            result.append(p)
        errors[m] = r.model_error_bars()
        means[m] = r.mean
        stds[m] = r.std
        results.append(result)

    df = pd.DataFrame(results)
    df.columns = ["model", "beta", "beta_ci", "bic", "p_value", "alpha", "beta_", "gamma", "delta"]
    df.set_index("model", inplace=True)

    return df, means, stds, errors


def get_best_model_name(results, highest_bic_only=False):
    best_model = sorted([(results.loc[m]['bic'], m) for m in pure_model], key=lambda x: x[0])[0][1]
    if not highest_bic_only:
        # now we test against the same with beta fixed:
        m_fixed = best_model + "_beta_fixed"
        delta_bic = results.loc[m_fixed].bic - results.loc[best_model].bic
        if 0 <= delta_bic < 6:
            # inconclusive, so we say that it is linear
            best_model = m_fixed
        elif delta_bic < 0:
            # linear is better
            best_model = m_fixed
        else:
            # best_model is actually the best
            best_model = best_model
    return best_model


def get_best_model(results, **kwargs):
    best_model = get_best_model_name(results, **kwargs)
    best_model_values = results.loc[best_model]
    alpha, beta, delta, gamma = \
        best_model_values.alpha, best_model_values.beta, best_model_values.delta, best_model_values.gamma
    gamma = np.nan if gamma == '' else gamma
    delta = np.nan if delta == '' else delta
    return ScalingLaw(best_model, alpha, beta, delta, gamma)


class ScalingLaw:

    def __init__(self, model_name, alpha, beta, delta, gamma):
        self.alpha = alpha
        self.beta = beta
        self.delta = delta
        self.gamma = gamma
        self.model_name = model_name

        if "lognormal" in model_name:
            self.std_internal = ScalingLaw.std_lognormal
            self.z_score_internal = ScalingLaw.z_score_lognormal
        else:
            self.std_internal = ScalingLaw.std_gaussian
            self.z_score_internal = ScalingLaw.z_score_gaussian

    def mean(self, x):
        return self.alpha * np.power(x, self.beta)

    def std(self, x, sigmas=1.0):
        return self.std_internal(self.alpha, self.beta, self.delta, self.gamma, x, sigmas=sigmas)

    def z_score(self, x, y):
        return self.z_score_internal(x, y, self.alpha, self.beta, self.delta, self.gamma)

    def rank_cities(self, df):
        data_xy = np.array(df.population.values) * 1.0, np.array(df.crime.values) * 1.0
        per_capita = 100000 * data_xy[1] / data_xy[0]  # crime per 100,000 inhabitants
        z_score = self.z_score(*data_xy)

        # let's maintain data order
        index_sequence = range(len(data_xy[0]))
        cities_index_z_score = zip(index_sequence, z_score)

        # z-score ranking
        cities_index_z_score_ranking = sorted(
            cities_index_z_score, key=lambda x: x[1], reverse=True)  # highest in the beginning
        cities_index_z_score_ranking = sorted(
            zip(*zip(*cities_index_z_score_ranking), index_sequence), key=lambda x: x[0])

        # per capita ranking
        cities_index_per_capita = sorted(
            zip(index_sequence, per_capita), key=lambda x: x[1], reverse=True)  # highest in the beginning
        cities_index_per_capita_ranking = sorted(
            zip(*zip(*cities_index_per_capita), index_sequence), key=lambda x: x[0])

        ranking_per_capita = list(zip(*cities_index_per_capita_ranking))[2]
        ranking_z_score = list(zip(*cities_index_z_score_ranking))[2]

        df['per_capita'] = per_capita
        df['z_score'] = z_score
        df['per_capita_ranking'] = ranking_per_capita
        df['adjusted_ranking'] = ranking_z_score

        return df

    @staticmethod
    def std_lognormal(alpha, beta, delta, gamma, x, sigmas=1.0):
        log_x = np.log10(x)
        mean = alpha * np.power(x, beta)
        var_log = np.log10(1 + gamma * np.power(mean, (delta - 2)))
        mean_log = np.log10(alpha) + beta * log_x - var_log / 2
        std_log = np.sqrt(var_log)
        return [10 ** (mean_log - sigmas * std_log), 10 ** (mean_log + sigmas * std_log)]

    @staticmethod
    def std_gaussian(alpha, beta, delta, gamma, x, sigmas=1.0):
        mean_y = alpha * np.power(x, beta)
        std = gamma * np.power(mean_y, delta)
        return [mean_y - sigmas * std, mean_y + sigmas * std]

    @staticmethod
    def z_score_gaussian(x, y, alpha, beta, delta, gamma):
        mean = alpha*(x**beta)
        std = gamma*np.power(mean, delta)
    #     std = np.sqrt(gamma*np.power(mean, delta))
        z_score = (y - mean)/std
        return z_score

    @staticmethod
    def z_score_lognormal(x, y, alpha, beta, delta, gamma):
        x_, y_ = np.log(x), np.log(y)
        x = np.exp(x_)
        mean = alpha*np.power(x, beta)
        var_log = np.log(1 + gamma * np.power(mean, (delta - 2)))
        mean_log = np.log(alpha) + beta*x_ - var_log/2
        z_score = (y_ - mean_log)/var_log
    #     z_score = (y_ - mean_log)/np.sqrt(var_log)
        return z_score
