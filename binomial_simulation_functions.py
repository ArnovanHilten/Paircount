import numpy as np


def simulation_gusev(num_patients=100, num_features=100, n=2, p=0.3, random_seed=42, h2=0.75, yseed=10):
    """
    Simulation phenotype =  h^2 * g + noise

    based on blog post by Sasha Gusev http://sashagusev.github.io/2015-12/notes-on-GCTA.html
    :param num_patients: number of patients
    :param num_features: number of features
    :param n: genotype 0 1 2 with n=2
    :param p: second parameter of binominal distribibution
    :param random_seed: random seed
    :param h2: heritability
    :param yseed: seed needs to be consistent for val test
    :return:
    """
    np.random.seed(random_seed)
    geno = np.zeros([num_patients, num_features])

    for k in range(num_features):
        geno[:, k] = np.random.binomial(n, p, num_patients)

    geno = (geno - np.matmul(np.ones((geno.shape[0], 1)), np.reshape(np.mean(geno, axis=0), (1, -1)))) / (
        np.matmul(np.ones((geno.shape[0], 1)), np.reshape(np.std(geno, axis=0), (1, -1))))

    A = np.dot(geno, np.transpose(geno)) / num_features

    # 1. SNP effects
    np.random.seed(yseed)
    u = np.random.randn(num_features)
    np.random.seed(random_seed)
    # 2. genetic value
    g = np.dot(geno, u)
    g = (g - np.mean(g)) / np.std(g)

    # 3. add environmental noise
    pheno = np.sqrt(h2) * g + np.random.randn(num_patients) * np.sqrt(1 - h2)

    pheno = (pheno - np.mean(pheno)) / np.std(pheno)

    return geno, pheno, u


def simulation_binomial_linked(num_patients=100, num_features=100,
                    ind_linked=[[[0, 1], [2, 2], [5, 1]], [[18, 2]], [[0, 1], [3, 2]], [[4, 1], [7, 1]]], n=2, p=0.3,
                    random_seed=42):
    """Simple non-linear simulation. Binominal distribution, phenotype is defined by ind_linked. If the combiniations
     in ind_linked are there a subject will get phenotype = 0 otherwise phenotype =0
     :param num_patients: number of patients
     :param num_features: number of input features
     :param ind_linked: list of lists with the combinations that are cause the phenotype.
     :param n: Parameter of the distribution, >= 0. Floats are also accepted, but they will be truncated to integers.
     :param p: Parameter of the distribution, >= 0
     :param random_seed: set random seed to get the same output for multiple calls
     :return: numpy array with simulated genotype, np.array with phenotype labels"""

    np.random.seed(random_seed)
    basis = np.zeros([num_patients, num_features])

    for k in range(num_features):
        basis[:, k] = np.random.binomial(n, p, num_patients)

    status = np.zeros(num_patients)
    for patient in range(num_patients):
        for linked in ind_linked:
            temp = np.zeros([len(linked)])
            i = 0
            for element in linked:
                if basis[patient, element[0]] == element[1]:
                    temp[i] = 1
                i += 1
            if np.min(temp) > 0:
                status[patient] = 1

    num_diseased = np.sum(status)
    print(("Created dataset[", num_patients, " x ", num_features, "] with", num_diseased, "diseased"))
    return basis, status


def simulation_gusev_linked(num_patients=100, num_features=100,
                 ind_linked=[[[0, 1], [2, 2], [5, 1]], [[18, 2]], [[0, 1], [3, 2]], [[4, 1], [7, 1]]], n=2, p=0.3,
                 random_seed=42, h2=0.75, i2=0.05):
    """Simulation with ind_linked. this simulation allows the definition of the heritability of the trait and the
    heritability of the interaction.

    Pheno =  genotype * h2 + interaction_effect * i2 + noise

    inter status: if the ind_linked combinations are there
    u: vector of effect for the interaction

    :param num_patients: number of patients
    :param num_features: number of input features
    :param ind_linked: list of lists with the combinations that are cause the phenotype.
    :param n: Parameter of the distribution, >= 0. Floats are also accepted, but they will be truncated to integers.
    :param p: Parameter of the distribution, >= 0
    :param random_seed: set random seed to get the same output for multiple calls
    :param h2: heritability of the additve part of the trait
    :param i2: heritability of the interactive part
    :return:  numpy array with simulated genotype, np.array with phenotype labels"""

    assert h2 + i2 < 1, "The total heritability can't be greater than 1 "

    np.random.seed(random_seed)
    geno = np.zeros([num_patients, num_features])

    for k in range(num_features):
        geno[:, k] = np.random.binomial(n, p, num_patients)

    inter_status = np.zeros(num_patients)
    for patient in range(num_patients):
        for linked in ind_linked:
            temp = np.zeros([len(linked)])
            i = 0
            for element in linked:
                if geno[patient, element[0]] == element[1]:
                    temp[i] = 1
                i += 1
            if np.min(temp) > 0:
                inter_status[patient] = 1

    geno = (geno - np.matmul(np.ones((geno.shape[0], 1)), np.reshape(np.mean(geno, axis=0), (1, -1)))) / (
        np.matmul(np.ones((geno.shape[0], 1)), np.reshape(np.std(geno, axis=0), (1, -1))))

    # 1. SNP effects
    u = np.random.randn(num_features)
    # 2. genetic value
    g = np.dot(geno, u)
    g = (g - np.mean(g)) / np.std(g)

    np.random.seed(random_seed + 1)
    u = np.random.randn(num_features)

    inter = np.dot(inter_status, u)
    # 3. add environmental noise
    pheno = np.sqrt(h2) * g + np.sqrt(i2) * inter + np.random.randn(num_patients) * np.sqrt(1 - i2 - h2)
    pheno = (pheno - np.mean(pheno)) / np.std(pheno)

    return geno, pheno


def simulation_random_feature_interaction(num_patients=100, num_features=100,
                               frac=0.1, n=2, p=0.3,
                               random_seed=42, h2=0.75, i2=0.05, yseed=0):
    """
    Simulation where a fraction of features randomly interact

    :param num_patients: number of patients
    :param num_features: number of features
    :param frac: fraction of interacting features
    :param n: parameter of binominal distribibution
    :param p: second parameter of binominal distribibution
    :param random_seed: random seed for genotype
    :param h2: additive heritability
    :param i2: interactive heritability
    :param yseed: seed for random effect
    :return: genotype, phenotype, index of interaction, strength additive effects, strength  interactive effects
    """

    np.random.seed(yseed)
    geno = np.zeros([num_patients, num_features])
    num_interact = int(np.round(frac * num_features))
    indices_interact = np.random.randint(0, num_features,
                                         size=(num_interact, 2))  # TODO: features can interact with themselves!

    # create genotype
    np.random.seed(random_seed)
    for k in range(num_features):
        geno[:, k] = np.random.binomial(n, p, num_patients)

    # create vector with interacting features
    inter_status = np.zeros((num_patients, num_interact))
    for patient in range(num_patients):
        for i in range(num_interact):
            if (geno[patient, indices_interact[i, 0]] == 2) & (geno[patient, indices_interact[i, 1]] == 2):
                inter_status[patient, i] = 1

    # standardize genotype
    geno = (geno - np.matmul(np.ones((geno.shape[0], 1)), np.reshape(np.mean(geno, axis=0), (1, -1)))) / (
        np.matmul(np.ones((geno.shape[0], 1)), np.reshape(np.std(geno, axis=0), (1, -1))))

    # SNP effects
    np.random.seed(yseed)
    u1 = np.random.randn(num_features)
    # genetic value
    g = np.dot(geno, u1)
    g = (g - np.mean(g)) / np.std(g)

    np.random.seed(yseed + 1337)
    u2 = np.random.randn(num_interact)

    # interactive value
    inter = np.dot(inter_status, u2)
    inter = (inter - np.mean(inter)) / np.std(inter)

    # add environmental noise
    np.random.seed(random_seed)
    pheno = np.sqrt(h2) * g + np.sqrt(i2) * inter + np.random.randn(num_patients) * np.sqrt(1 - i2 - h2)

    # standardize genotype
    pheno = (pheno - np.mean(pheno)) / np.std(pheno)

    return geno, pheno, indices_interact, u1, u2


def simulation_addative_nonaddative_matrix(num_patients, num_features,
                                           frac, random_seed, h2, yseed, verbose=1, n=2):
    """
    Simulation with an additive matrix and non-additive matrix based on combinations of the additive matrix
    :param num_patients: number of patients
    :param num_features: number of features
    :param frac: fraction of features cannot be higher than
    :param n: genotype 0 1 2 with n=2
    :param random_seed: random seed
    :param h2: heritability
    :param yseed: seed needs to be consistent for val test
    :param verbose: print the heritability
    :return:
    """
    np.random.seed(yseed)
    geno = np.zeros([num_patients, num_features])
    num_interact = int(np.round(frac * num_features))
    q = 1

    # get interactions but without self interaction
    not_done = True
    while not_done:
        test = True
        indices_interact = np.zeros([num_interact, 2])
        i1 = np.arange(num_features)
        i2 = np.arange(num_features)
        for i in range(num_interact):
            indices_interact[i, 0] = np.random.choice(i1, replace=False)  # without replacement
            indices_interact[i, 1] = np.random.choice(i2, replace=False)
        for i in range(num_interact):
            if (indices_interact[i, 1] == indices_interact[i, 0]):  # check for self interaction
                test = False
        if test:
            not_done = False  # if self interaction test is passed then we are done
        else:
            q += 1
            print(q)
    indices_interact = indices_interact.astype(int)

    # create maf
    p = np.random.random(size=(num_features,)) * 0.45 + 0.05
    # create genotype
    np.random.seed(random_seed)
    for k in range(num_features):
        geno[:, k] = np.random.binomial(n, p[k], num_patients)
    np.random.seed(yseed)
    # create vector with interacting features
    interaction_matrix = np.zeros([num_patients, num_interact])
    for SNPin in range(num_interact):
        interaction_matrix[:, SNPin] = np.ceil(
            (geno[:, indices_interact[SNPin, 0]] * geno[:, indices_interact[SNPin, 1]]) / 2).astype(
            np.int)  # dominant wrt (1+2) /2 = r(1.5) -> 2

    # merge as one:
    total_matrix = np.concatenate((geno, interaction_matrix), axis=1)

    for feat in range(total_matrix.shape[1]):
        total_matrix[:, feat] = (total_matrix[:, feat] - np.mean(total_matrix[:, feat])) / np.std(total_matrix[:, feat])

    # standardize genotype
    # total_matrix3 = (total_matrix - np.matmul(np.ones((total_matrix.shape[0], 1)), np.reshape(np.mean(total_matrix, axis=0), (1, -1)))) / (
    #     np.matmul(np.ones((total_matrix.shape[0], 1)), np.reshape(np.std(total_matrix, axis=0), (1, -1))))
    # GRMatrix
    # GRM1 = np.dot(total_matrix[:, :num_features], np.transpose(total_matrix[:, :num_features], )) / num_features
    # GRM2 = np.dot(total_matrix[:, num_features:], np.transpose(total_matrix[:, num_features:])) / num_features

    # 1. SNP effects
    u = np.random.randn(num_features + num_interact)
    # 2. genetic value
    g = np.dot(total_matrix, u)
    g = (g - np.mean(g)) / np.std(g)

    # 3. add environmental noise
    np.random.seed(random_seed)
    pheno = np.sqrt(h2) * g + np.random.randn(num_patients) * np.sqrt(1 - h2)
    np.random.seed(yseed)
    pheno = (pheno - np.mean(pheno)) / np.std(pheno)

    geno = total_matrix[:, :num_features]
    inter = total_matrix[:, num_features:]

    additive_percentage = np.sum(np.dot(np.abs(total_matrix[:, :num_features]), np.abs(u[:num_features]))) / np.sum(
        np.dot(np.abs(total_matrix), np.abs(u)))

    nonadditive_percentage = np.sum(np.dot(np.abs(total_matrix[:, num_features:]), np.abs(u[num_features:]))) / np.sum(
        np.dot(np.abs(total_matrix), np.abs(u)))

    additive_h2 = h2 * additive_percentage
    nonadditive_h2 = h2 * nonadditive_percentage

    if verbose:
        # print("additive percentage of h2= " + str( additive_percentage))
        # print("non-additive percentage of h= " + str(nonadditive_percentage))
        print("total additive percentage = " + str(additive_h2))
        print("total non-additive percentage = " + str(nonadditive_h2))
        print("total noise = " + str(1 - h2))

    return geno, pheno, inter, indices_interact, u, nonadditive_h2, additive_h2 #  GRM1, GRM2


def simulation_addative_nonaddative_matrix_with_effect(num_patients, num_features,
                          frac, random_seed, h2, yseed, verbose=1, n=2, effect_lin=1):
    '''
    Simulation with an additive matrix and non-additive matrix based on combinations of additive
    :param num_patients: number of patients
    :param num_features: number of features
    :param frac: fraction of features cannot be higher than
    :param n: genotype 0 1 2 with n=2
    :param random_seed: random seed
    :param h2: heritability
    :param yseed: seed needs to be consistent for val test
    :param verbose: print the heritability
    :param effect_lin:effect size linear, between 0 and 1
    :return:
    '''
    np.random.seed(yseed)
    geno = np.zeros([num_patients, num_features])
    num_interact = int(np.round(frac * num_features))
    q = 1

    # get interactions but without self interaction
    not_done = True
    while not_done:
        test = True
        indices_interact = np.zeros([num_interact, 2])
        i1 = np.arange(num_features)
        i2 = np.arange(num_features)
        for i in range(num_interact):
            indices_interact[i, 0] = np.random.choice(i1, replace=False)  # without replacement
            indices_interact[i, 1] = np.random.choice(i2, replace=False)
        for i in range(num_interact):
            if (indices_interact[i, 1] == indices_interact[i, 0]):  # check for self interaction
                test = False
        if test:
            not_done = False  # if self interaction test is passed then we are done
        else:
            q += 1
            print(q)
    indices_interact = indices_interact.astype(int)

    # create maf
    p = np.random.random(size=(num_features,)) * 0.45 + 0.05
    # create genotype
    np.random.seed(random_seed)
    for k in range(num_features):
        geno[:, k] = np.random.binomial(n, p[k], num_patients)
    np.random.seed(yseed)
    # create vector with interacting features
    interaction_matrix = np.zeros([num_patients, num_interact])
    for SNPin in range(num_interact):
        interaction_matrix[:, SNPin] = np.ceil(
            (geno[:, indices_interact[SNPin, 0]] * geno[:, indices_interact[SNPin, 1]]) / 2).astype(
            np.int)  # dominant wrt (1+2) /2 = r(1.5) -> 2

    # merge as one:
    total_matrix = np.concatenate((geno, interaction_matrix), axis=1)

    for feat in range(total_matrix.shape[1]):
        total_matrix[:, feat] = (total_matrix[:, feat] - np.mean(total_matrix[:, feat])) / np.std(total_matrix[:, feat])

    # standardize genotype
    # total_matrix3 = (total_matrix - np.matmul(np.ones((total_matrix.shape[0], 1)), np.reshape(np.mean(total_matrix, axis=0), (1, -1)))) / (
    #     np.matmul(np.ones((total_matrix.shape[0], 1)), np.reshape(np.std(total_matrix, axis=0), (1, -1))))

    # GRMatrix
    GRM1 = np.dot(total_matrix[:, :num_features], np.transpose(total_matrix[:, :num_features], )) / num_features
    GRM2 = np.dot(total_matrix[:, num_features:], np.transpose(total_matrix[:, num_features:])) / num_features

    effect_nonlin = ((num_interact + num_features) - effect_lin * num_features) / num_interact

    print("linear effects = " + str(effect_lin))
    print("nonlinear effects = " + str(effect_nonlin))

    # 1. SNP effects
    u1 = np.random.randn(num_features) * np.sqrt(effect_lin)
    u2 = np.random.randn(num_interact) * np.sqrt(effect_nonlin)  # TODO: use samplsize to make u3 simga 1?

    u = np.concatenate((u1, u2))

    u = (u - np.mean(u)) / np.std(u)

    # 2. genetic value
    g = np.dot(total_matrix, u)
    g = (g - np.mean(g)) / np.std(g)

    # 3. add environmental noise
    np.random.seed(random_seed)
    pheno = np.sqrt(h2) * g + np.random.randn(num_patients) * np.sqrt(1 - h2)
    np.random.seed(yseed)
    pheno = (pheno - np.mean(pheno)) / np.std(pheno)

    geno = total_matrix[:, :num_features]
    inter = total_matrix[:, num_features:]

    additive_percentage = np.sum(np.dot(np.abs(total_matrix[:, :num_features]), np.abs(u[:num_features]))) / np.sum(
        np.dot(np.abs(total_matrix), np.abs(u)))

    nonadditive_percentage = np.sum(np.dot(np.abs(total_matrix[:, num_features:]), np.abs(u[num_features:]))) / np.sum(
        np.dot(np.abs(total_matrix), np.abs(u)))

    additive_h2 = h2 * additive_percentage
    nonadditive_h2 = h2 * nonadditive_percentage

    if verbose:
        # print("additive percentage of h2= " + str( additive_percentage))
        # print("non-additive percentage of h= " + str(nonadditive_percentage))
        print("total additive percentage = " + str(additive_h2))
        print("total non-additive percentage = " + str(nonadditive_h2))
        print("total noise = " + str(1 - h2))

    return geno, pheno, inter, indices_interact, u, GRM1, GRM2, nonadditive_h2, additive_h2
