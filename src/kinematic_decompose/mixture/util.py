import scipy
import numpy as np

MAX_RADIUS = 10

def hist_bin_fd(x):
    iqr = np.subtract(*np.percentile(x, [75, 25]))
    return 2.0 * iqr * x.size ** (-1.0 / 3.0)

def get_energy_criterion(pot, galaxy):
    if pot is None:
        import warnings
        warnings.warn(
            "pot == None, using the method from Zana et al. 2022",
            UserWarning,
            stacklevel=2
        )
        return get_Ecut(galaxy.s['eoemin'], galaxy.s['mass'])
    eps = galaxy.properties.get('eps', 1)
    R_max = min(5*galaxy.s.r50/1.414, MAX_RADIUS)
    
    x = y = np.arange(eps, R_max, eps)
    points = np.column_stack([x,y,np.zeros(len(x))])
    R = np.linalg.norm(points, axis=1)
    potential = pot.potential(points) 
    dpot_dR = np.gradient(potential/pot.potential([1e-2,1e-2,1e-2]), R, edge_order=2)
    d2pot_dR2 = np.gradient(dpot_dR, R, edge_order=2)
    curvature = np.abs(d2pot_dR2) / (1 + dpot_dR**2)**1.5
    rcut = R[np.argmax(curvature)]
     
    e = galaxy.s['eoemin']
    r = galaxy.s['R'] 

    r_median, bin_edges, _ = scipy.stats.binned_statistic(
        e, r, statistic='median', bins=50, range=[-1, -0.]
    )
    ebin = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    valid_mask = np.isfinite(r_median)
    r_median, ebin = r_median[valid_mask], ebin[valid_mask]

    mask = (r_median < MAX_RADIUS) & (r_median > eps)
    if np.sum(mask) < MAX_RADIUS: mask = (r_median > 0)
    x, y = ebin[mask], r_median[mask]
    y = np.maximum.accumulate(y)
    y, idx = np.unique(y, return_index=True)
    x = x[idx]

    f_r = scipy.interpolate.PchipInterpolator(y, x, extrapolate=True)
    #f_e = PchipInterpolator(x, y, extrapolate=True)

    ecut = f_r(rcut)

    return ecut 


def get_Ecut(eb, masses, nbins = 25,M_bin=400,m_bin=80,toll=1.5,shrink=2,Mmin = 0.05,Emin = -0.9):
    
    if len(eb)<100:
        print('Ecut = ', 0)
        return 0

    #Fix the number of bin as a function of Npart
    NbinMax = max(min(int(0.5*np.sqrt(len(eb))), M_bin), m_bin)


    #This is to exclude the outer tail of bound particles
    M_E = np.quantile(eb, 0.9)
    #m_E = np.min(eb)
    m_E = np.quantile(eb, 0.01)
    Ecut, E_val = FindMin(eb, m_E, M_E, nbins)


    #If no minimum is found or the only minimum is too close to -1 (Maybe a GC?)
    if len(Ecut)==0 or (len(Ecut)==1 and Ecut<Emin):


        M_E = np.max(eb)
        Ecut, E_val = FindMin(eb, m_E, M_E, nbins)
        Ecut = Ecut
    #If one or none minima are found
    if len(Ecut)<=1:
        D = (M_E-m_E)/float(nbins)
        #Avoid the following loop
        nbins = NbinMax+1
    else:
        D = (M_E-m_E)/float(nbins)
        lb = Ecut-(toll*D)
        rb = Ecut+(toll*D)

    #-------

    while nbins < NbinMax:
        nbins = shrink*nbins
        D = D/shrink
        pos_E_refined, val_refined = FindMin(eb, m_E, M_E, nbins)
        EcutTEMP = []
        E_valTEMP = []
        for i, v in enumerate(E_val):
            pTEMP = pos_E_refined[(pos_E_refined<=rb[i])*(pos_E_refined>=lb[i])]
            vTEMP = val_refined[(pos_E_refined<=rb[i])*(pos_E_refined>=lb[i])]
            if len(pTEMP)>0:
                #A rifened position and value for each original minimum is stored. 
                #The value of the minima is summed to the original ones to avoid strange local minima 
                EcutTEMP.append(pTEMP[np.argmin(vTEMP)])
                E_valTEMP.append(v+np.min(vTEMP))

        Ecut = np.array(EcutTEMP)
        E_val = np.array(E_valTEMP)



        if len(Ecut)<=1:
            break

        lb = Ecut-(toll*D)
        rb = Ecut+(toll*D)
    
    #If no energy cut is found		
    if len(Ecut)==0:
        Ecut = 0
    else:
        #Try to avoid strange nuclear minima with low mass if there are better alternatives
        rel_filt = [bool((np.sum(masses[eb<E])/np.sum(masses)>=Mmin)+(E>=Emin)) for E in Ecut]
        if len(Ecut[rel_filt])==0:
            Ecut = Ecut[np.argmin(E_val)]
        else:
            Ecut = Ecut[rel_filt][np.argmin(E_val[rel_filt])]
        Ecut = RefineMin(eb, Ecut, D, (M_E-m_E)/NbinMax, shrink)
    
    return Ecut

def FindMin(q, m_E, M_E, nbins):
    #Minimum number of particles to perform a reliable Jcirc decomposition
    if len(q)>=1e4:
        Npart_min=1000
    elif 1e3<=len(q)<1e4:
        Npart_min=100
    else:
        Npart_min=10
    
    MinPart = max(Npart_min, 0.01*len(q))
    arr = q[(q>=m_E)*(q<=M_E)]
    #Build the histogram
    hist = np.histogram(arr, bins=np.linspace(m_E, M_E, nbins))

    #Evaluate the increment on both sides A
    diff = hist[0][1:]-hist[0][:-1]
    left = diff[:-1]
    right = diff[1:]
    #Find the minima
    id_E = np.where(((left<0)*(right>=0))+((left<=0)*(right>0)))

    #C
    R_part = np.array([np.sum(hist[0][i+1:]) for i in id_E[0]])
    id_E = id_E[0][R_part>MinPart]
    id_E_flag = [True]*len(id_E)
    
    
    #B
    for i, ids in enumerate(id_E):
        if len(hist[0])>ids+3:
            id_E_flag[i] *= hist[0][ids+3]>hist[0][ids+1]
        if ids > 0:
            id_E_flag[i] *= hist[0][ids-1]>hist[0][ids+1]

    id_E = id_E[id_E_flag]
    #Return the central position of the bins
    return 0.5*(hist[1][id_E+2]+hist[1][id_E+1]), hist[0][id_E+1]


def RefineMin(q, Vmin, D, Dmin, shrink):    
    arr=[]
    if D<=Dmin:
        if len(q)>=1e4:
            coe=0.5
        elif 1e3<=len(q)<1e4:
            coe=2
        else:
            coe=2.5
        while len(arr)==0:
            m_E = Vmin-coe*D 
            M_E = Vmin+coe*D
            arr = q[(q>=m_E)*(q<=M_E)]
            coe=coe+0.2
        Vmin = np.median(arr)

    while D > Dmin:
        if len(q)>=1e3:
            coe=1.5
        else:
            coe=4
        m_E = max(Vmin-coe*D, q.min()+D) 
        M_E = Vmin+coe*D
        D = D/shrink
        arr = q[(q>=m_E)*(q<=M_E)]
        hist = np.histogram(arr, bins=np.arange(m_E, M_E, D))
        hist_min=(hist[0][np.where(hist[0]!=0)]).min()
        pid = np.where(hist[0]==hist_min)[0][0]
        arr=arr[(arr>=hist[1][pid])*(arr<=(hist[1][pid+1]))]
        #Get the energy as the median within the selected bin
        Vmin = np.median(arr)

    return Vmin

def JEHistogram(E, eps, n_E=20, n_eps=30, seed=42):
    """
    Abadi-like decomposition with local eps symmetry.

    Assumptions:
    - All counter-rotating particles (eps < 0) belong to spheroid.
    - For each energy bin and each eps bin, the same number of
      co-rotating particles are added to spheroid to ensure
      local symmetry around eps=0.
    - Remaining co-rotating particles belong to disk.
    """
    rng = np.random.default_rng(seed)
    N = len(E)
    sph = np.zeros(N, dtype=bool)

    # Energy bins
    E_edges = np.linspace(E.min(), E.max(), n_E + 1)

    for i in range(n_E):
        e_mask = (E >= E_edges[i]) & (E < E_edges[i + 1])
        idx_E = np.flatnonzero(e_mask)
        if idx_E.size == 0:
            continue

        eps_E = eps[idx_E]

        # Symmetric eps bins
        eps_max = np.max(np.abs(eps_E))
        if eps_max == 0:
            sph[idx_E] = True
            continue

        eps_edges = np.linspace(-eps_max, eps_max, n_eps + 1)
        bin_id = np.searchsorted(eps_edges, eps_E, side="right") - 1
        bin_id = np.clip(bin_id, 0, n_eps - 1)

        mid = n_eps // 2

        # Loop over symmetric eps bins
        for k in range(mid):
            neg_bin = mid - 1 - k
            pos_bin = mid + k

            idx_neg = idx_E[bin_id == neg_bin]
            idx_pos = idx_E[bin_id == pos_bin]

            if idx_neg.size == 0:
                continue

            # All counter-rotating are spheroid
            sph[idx_neg] = True

            if idx_pos.size == 0:
                continue

            n_sel = min(idx_neg.size, idx_pos.size)
            chosen = (
                idx_pos if idx_pos.size == n_sel
                else rng.choice(idx_pos, n_sel, replace=False)
            )
            sph[chosen] = True

        # Optional: eps ~ 0 bin → spheroid
        sph[idx_E[bin_id == mid]] = True

    disk = ~sph
    return sph, disk
