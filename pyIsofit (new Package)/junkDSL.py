if step == 4:
    gmod = Model(dsl)
    c = []
    result = []
    for i in range(len(keyPressures)):
        pars = Parameters()
        pars.add('q1', value=q1_init, min=0)
        pars.add('q2', value=q2_init, min=0)
        pars.add('b2', value=b2_init, min=0)
        pars.add('b1', value=b1_init, min=0) #KH = b*q

        results = gmod.fit(y[i], pars, x=x[i], method=meth)
        cee = [results.values['q1'], results.values['q2'], 
                    results.values['b1'], results.values['b2']]
        c.append(cee)
        result.append(results)

        del results
        del pars

            
    #allocating variables
    qmax1 = [param[0] for param in c]
    qmax2 = [param[1] for param in c]
    b1 = [param[2] for param in c]
    b2 = [param[3] for param in c]
    qtot = [param[0] + param[1] for param in c]
    t = temps

    
    # Checking r squared of fits
    r_sq = [r2(x[i], y[i], dsl, c[i]) for i in range(len(keyPressures))]
    se = [mse(x[i], y[i], dsl, c[i]) for i in range(len(keyPressures))]
    
    #Displaying results
    df_result = pd.DataFrame(list(zip(t, qmax1, qmax2, b1, b2, r_sq, se)), 
                    columns=['Temp(K)','qmax1 (mmol/g)',
                            'qmax2 (mmol/g)','b1 (1/bar)', 'b2 (1/bar)' ,'R sq', 'mse'])

    display(pd.DataFrame(df_result))
    
    print(bold + "===============================================================================")
    print(bold + "===============================================================================")

    xaxis = 'pressure [bar]'
    yaxis = 'uptake mmol/g'

    ##### Plotting results #####
    plt.figure(figsize=(8, 6))
    plt.title(compname)
    if logplot == True:
        plt.xscale("log")
        plt.yscale("log")
    plt.xlabel(xaxis)
    plt.ylabel(yaxis)
    plt.tick_params(**tick_style)
    
    for i in range(len(keyPressures)):
        plt.plot(x[i], result[i].best_fit, '-', color = colours[i], 
                    label="{temps} °C Fit".format(temps=temps[i]))
        plt.plot(x[i], y[i], 'ko', color = '0.75', 
                    label="Data at {temps} °C".format(temps=temps[i]))

    plt.grid()
    plt.legend()