import numpy as np
import pandas as pd
import numpy.random as npr
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from scipy import sparse, stats
from scipy.special import expit
import scipy.stats


class gwas_simulated_data(object):
    # Reference:
    # https://github.com/raquelaoki/ParKCa/blob/master/src/datapreprocessing.py

    def __init__(self, n_units=10000, n_causes=100, seed=4, pca_path='data//tgp_pca2.txt', prop_tc=0.1,
                 true_causes=None):
        self.n_units = n_units
        self.n_causes = n_causes
        if true_causes is None:
            self.true_causes = np.max([1, int(n_causes * prop_tc)])
        else:
            self.true_causes = true_causes
        self.confounders = self.n_causes - self.true_causes
        self.seed = seed
        self.pca_path = pca_path
        self.S = np.loadtxt(self.pca_path, delimiter=',')
        self.prop_tc = prop_tc
        print('GWAS simulated data initialized!')
        print('... ', self.true_causes, 'true causes and ', self.confounders, ' confounders')

    def generate_samples(self):
        """
        Input:
        n_units, n_causes: dimentions
        snp_simulated datasets
        y: output simulated and truecases for each datset are together in a single matrix
        Note: There are options to load the data from vcf format and run the pca
        Due running time, we save the files and load from the pca.txt file
        """
        G0, lambdas = self.sim_genes_TGP(D=3)
        G1, tc, y01, y, col, group = self.sim_dataset(G0, lambdas, self.prop_tc)
        # G, col = self.add_colnames(G1,tc)
        del G0
        return G1, y, y01, col, tc, group

    def sim_genes_TGP(self, D):
        """
        #Adapted from Deconfounder's authors
        generate the simulated data
        input:
            - Fs, ps, n_hapmapgenes: not adopted in this example
            - n_causes = integer
            - n_units = m (columns)
            - S: PCA output n x 2
        """
        np.random.seed(self.seed)
        S = expit(self.S)
        Gammamat = np.zeros((self.n_causes, 3))
        Gammamat[:, 0] = 0.2 * npr.uniform(size=self.n_causes)  # 0.45
        Gammamat[:, 1] = 0.2 * npr.uniform(size=self.n_causes)  # 0.45
        Gammamat[:, 2] = 0.05 * np.ones(self.n_causes)
        S = np.column_stack((S[npr.choice(S.shape[0], size=self.n_units, replace=True),], \
                             np.ones(self.n_units)))
        F = S.dot(Gammamat.T)
        # it was 2 instead of 1: goal is make SNPs binary
        G = npr.binomial(1, F)
        # unobserved group
        lambdas = KMeans(n_clusters=3, random_state=123).fit(S).labels_
        # sG = sparse.csr_matrix(G)
        return G, lambdas

    def sim_dataset(self, G0, lambdas, prop_tc):
        """
        calculate the target Y based on the simulated dataset
        input:
        G0: level 0 data
        lambdas: unknown groups
        n_causes and n_units: int, dimensions of the dataset
        output:
        G: G0 in pandas format with colnames that indicate if its a cause or not
        tc: causal columns
        y01: binary target
        """
        np.random.seed(self.seed)
        tc_ = npr.normal(loc=0, scale=0.5 * 0.5, size=self.true_causes)
        tc = np.hstack((tc_, np.repeat(0.0, self.confounders)))  # True causes
        tau = stats.invgamma(3, 1).rvs(3, random_state=99)
        sigma = np.zeros(self.n_units)
        sigma = [tau[0] if lambdas[j] == 0 else sigma[j] for j in range(len(sigma))]
        sigma = [tau[1] if lambdas[j] == 1 else sigma[j] for j in range(len(sigma))]
        sigma = [tau[2] if lambdas[j] == 2 else sigma[j] for j in range(len(sigma))]
        y0 = np.array(tc).reshape(1, -1).dot(np.transpose(G0))
        l1 = lambdas.reshape(1, -1)
        y1 = (np.sqrt(np.var(y0)) / np.sqrt(0.4)) * (np.sqrt(0.4) / np.sqrt(np.var(l1))) * l1
        e = npr.normal(0, sigma, self.n_units).reshape(1, -1)
        y2 = (np.sqrt(np.var(y0)) / np.sqrt(0.4)) * (np.sqrt(0.2) / np.sqrt(np.var(e))) * e
        y = y0 + y1 + y2
        p = 1 / (1 + np.exp(y0 + y1 + y2))

        y01 = np.zeros(len(p[0]))
        y01 = [npr.binomial(1, p[0][i], 1)[0] for i in range(len(p[0]))]
        y01 = np.asarray(y01)
        G, col = self.add_colnames(G0, tc)
        y = y0 + y1 + y2

        prop = []
        for i in col:
            prop.append(np.sum(G.iloc[i]) / G.shape[0])

        print('... Treatments: ', len(col), ' proportions ', prop)
        print('... Confounders: ', G.shape[1] - len(col))
        print('... Target (y) :', np.sum(y01) / len(y01))
        print('... Sample Size:', G.shape[0])
        print(' Data Simulation Done!')
        return G, tc, y01, y, col, tau

    def add_colnames(self, data, truecauses):
        """
        from matrix to pandas dataframe, adding colnames
        """
        colnames = []
        causes = 0
        noncauses = 0
        columns = []
        for i in range(len(truecauses)):
            if truecauses[i] != 0:
                colnames.append('causal_' + str(causes))
                causes += 1
                columns.append(i)
            else:
                colnames.append('noncausal_' + str(noncauses))
                noncauses += 1

        data = pd.DataFrame(data)
        data.columns = colnames
        return data, columns


class copula_simulated_data(object):
    # Reference:
    # https://github.com/JiajingZ/CopulaSensitivity/blob/CopSens/simulation/GaussianT_BinaryY_nonlinearYT/GaussianT_BinaryY_nonlinearYT_RR.R
    # adapted from R to python
    def __init__(self, k=4, s=10, B=[2, 0.5, -0.4, 0.2], gamma=2.8, sigma2_t=1, sigma2_y=1, tau_l=[3, -1, 1, -0.06],
                 tau_nl=[-4], n=10000, seed=10):

        self.k = k  # number of treatments
        self.s = s  # number of confounders
        self.B = B  # ?
        self.gamma = gamma  # if either B or gamma = 0, there is no confounding
        self.sigma2_t = sigma2_t  # variance on treatments
        self.sigma2_y = sigma2_y  # variance on outcomes

        self.tau_l = tau_l  # linear effect coef
        self.tau_nl = tau_nl  # non linear effect coef
        self.coef_true = np.concatenate([tau_l, tau_nl], axis=0)
        self.n = n  # sample size
        self.seed = seed
        print('Copula simulated data initialized!')

    def g_yt(self, t, tau_l, tau_nl, ind=2):
        """
        t: t is n by k matrix
        outputs y
        #always make the treatment 2 modified and squared treatment 1
        """
        t[:, ind] = [item if item > 0 else 0.7 * item for item in t[:, ind]]
        y = t.dot(tau_l)
        for i, item in enumerate(tau_nl):
            y = y + pow(t[:, i], 2) * item
        return y

    def generate_samples(self):
        np.random.seed(self.seed)
        u = np.random.normal(loc=0, scale=1, size=self.n * self.s).reshape(self.n, self.s)
        if self.s > 1:
            pca = PCA(n_components=1)
            u1 = pca.fit_transform(u)
        else:
            u1 = u
        tr = np.repeat(u1, self.k).reshape(self.n, self.k) * self.B
        tr = tr + np.random.normal(loc=0, scale=pow(self.sigma2_t, 2), size=self.n * self.k).reshape(self.n, self.k)

        y_continuous = self.g_yt(tr, self.tau_l, self.tau_nl) + (u1 * self.gamma).reshape(self.n, ) + np.random.normal(
            loc=0, scale=self.sigma2_y, size=self.n)
        y_binary = [1 if item > y_continuous.mean() else 0 for item in y_continuous]  # very well balanced

        tr = pd.DataFrame(tr, columns=['t1', 't2', 't3', 't4'])
        print('... Treatments:', tr.shape)
        print('... Confounders:', u.shape)
        print('... Target (y):', np.sum(np.array(y_binary)) / len(y_binary))

        X = np.concatenate([tr.values, u], 1)
        print('Data Simulation Done!')
        true_coef = self.get_true_coefs()[0]
        true_coef = np.array(true_coef)[0]
        return X, np.array(y_continuous), np.array(y_binary), list(range(tr.shape[1])), true_coef

    def get_true_coefs(self):
        aux1 = np.linalg.solve(np.array(self.B).reshape(self.k, 1) * (
            np.transpose(self.B).reshape(1, self.k)) + self.sigma2_t * np.identity(self.k), np.identity(self.k))
        B = np.matrix(self.B).reshape(1, self.k)
        coef_mu_u_t = B.dot(aux1)

        # theoretical values #
        sigma_u_t = np.sqrt(1 - B * aux1 * np.transpose(B))[0, 0]
        sigma_ytilde_t = (np.sqrt(pow(self.gamma, 2) * pow(sigma_u_t, 2) + self.sigma2_y))  # [0,0]
        sigma_ytilde_t_do = np.sqrt(pow(self.gamma, 2) + self.sigma2_y)

        # true Treatment effect #
        t_choice = np.identity(self.k)
        t2 = np.matrix(np.zeros(self.k))
        ytilde_mean_do = self.g_yt(np.array(np.concatenate([t_choice, t2], axis=0)), self.tau_l, self.tau_nl)
        y_mean_do = scipy.stats.norm.cdf(ytilde_mean_do / sigma_ytilde_t_do)
        effect_true = y_mean_do[0:4] / y_mean_do[4]

        # true treatment effect bias #
        # 5 x 4 , 4 x 1 , 1 x 1
        ytilde_mean_do_bias = np.array(
            (np.concatenate([t_choice, t2], axis=0).dot(np.transpose(coef_mu_u_t))) * self.gamma)

        # true observed treatment effect #
        ytilde_mean_obs = ytilde_mean_do.reshape(self.k + 1, 1) + ytilde_mean_do_bias
        y_mean_obs = scipy.stats.norm.cdf(ytilde_mean_obs / sigma_ytilde_t)
        effect_obs = (y_mean_obs[0:4] / y_mean_obs[4]).reshape(1, self.k)
        # print('Start: Copula True Treatment Effects')
        # print("... True effect", effect_true)
        # print("... True obs effect", effect_obs)

        # print('\n... Binary Nonlinear:')
        # print("B: True treat. effect", ytilde_mean_do_bias.reshape(1,k+1))
        # print("... B: True treat. obs effect", ytilde_mean_obs.reshape(1, self.k + 1))

        # print('\n... Continuous Nonlinear')
        # true treatment effect #
        effect_true_c = self.g_yt(t_choice, self.tau_l, self.tau_nl) - self.g_yt(t2, self.tau_l, self.tau_nl)
        print(effect_true_c)
        # true treatment effect bias #
        effect_bias_c = ((t_choice.dot(np.transpose(coef_mu_u_t))) * self.gamma).reshape(1, self.k)
        print(effect_bias_c)
        # true observed treatment effect #
        effect_obs_c = effect_true_c.reshape(1, self.k) + effect_bias_c
        print(effect_obs_c)
        # print("C: True treat. effect", effect_bias_c)
        # print("... C: True treat. obs effect", effect_obs_c)
        # return effect_true, effect_obs, ytilde_mean_obs.reshape(1, self.k + 1), effect_obs_c
        return effect_bias_c

    def print_equation(self):
        eq = 'g(T)='
        t = ['T' + str(i) for i in range(self.k)]
        t[self.ind] = t[self.ind] + 'I(' + t[self.ind] + '>0)+0.7' + t[self.ind] + 'I(' + t[self.ind] + '<0)'

        if self.k < len(self.coef_true):
            nonlinear = len(self.coef_true) - self.k
            for item in range(nonlinear):
                t.append(t[item] + '*' + t[item])

        for i, item in enumerate(t):
            if self.coef_true[i] > 0:
                coef = '+' + str(round(self.coef_true[i]))
            else:
                coef = str(round(self.coef_true[i]))
            eq = eq + coef + item
        print('\nEquation:')
        print(eq)


class ihdp_data(object):
    # source code: https://github.com/AMLab-Amsterdam/CEVAE.git
    def __init__(self, id=1, path='/content/CEVAE/datasets/IHDP/'):
        data = pd.read_csv(path + 'csv/ihdp_npci_' + str(id) + '.csv', sep=',', header=None)
        columns = ['treatment', 'y_factual', 'y_cfactual', 'mu0', 'mu1', 'x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8',
                   'x9', 'x10', 'x11', 'x12', 'x13', 'x14', 'x15', 'x16', 'x17', 'x18', 'x19', 'x20', 'x21', 'x22',
                   'x23', 'x24', 'x25']
        data.columns = columns
        self.data = data
        print('IHCP initilized!')

    def generate_samples(self):
        X = self.data.drop(['y_factual', 'y_cfactual', 'mu0', 'mu1'], axis=1)
        y = self.data['y_factual'].values
        col = [0]
        tc = self.data['mu1'].mean() - self.data['mu0'].mean()
        return X, y, col, tc
