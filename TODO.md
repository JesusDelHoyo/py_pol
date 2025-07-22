# Todo


## After Beta

* https://py-pol.readthedocs.io/en/master/readme.html (Extracting information from Mueller matrices): print(m1.parameters)
* get_all() for Mueller
* Finish or remove help() function
* Check again that https://py-pol.readthedocs.io/en/master/installation.html works the git clone
* 3. Usage. See other examples and see if some more information is required.
* Reorder modules in https://py-pol.readthedocs.io/en/master/py_pol.html ?
    x diattenuator_linear .. math:: J = [p_1, 0; 0, p_2]’.
    x half_waveplate  .. math:: (lambda/2).
    x retarder_material phase=2∗π    rac{(ne-no)*d}{/lambda}.
    x general_charac_angles: References
    x charac_angles: References
    - filter_purify_number. Tiene un todo. ¿dejar o quitar?
    - filter_purify_threshold Tiene un todo. ¿dejar o quitar?
    -  help()[source]   todo.
    - is_diattenuator :math:`M = M^T and M `
    - is_pure: todo
    x is_retarder .. math
    - covariance_matrix Note:          :math:`H=0.25sum(m[i,j],kronleft[S(i),S(j)^{*} ight].`
    x diattenuator_linear :  with q=p1∗∗2,r=p2∗∗2.. Aquí la nota está bien en las de arriba queda peor.
    - mirror . make better warning
    x retarder_azimuth_ellipticity_from_vector: razimuth -> azimuth
    -py_pol.stokes module
        x degree_circular_polarization  P=rac{s3)}{s0}.
        x degree_linear_polarization P= rac{sqrt(s1**2+s2**2)}{s0}.
        x delay .. math:: delta_2 - delta_1.
        x delta: .. math:: delta_2 - delta_1.
        x from_Jones improve equations.
    4.1.7. py_pol.utils module
        - azimuth_elipt_2_charac_angles cos(2lpha)=cos(2az)∗cos(2el) tan(δ)=
                - Aquí references está como Parametesr, etc.
                - Returns todo en línea
        - charac_angles_2_azimuth_elipt -- idem
        - extract_azimuth_elipt -.- ecuaciones

* Tutorials
    - Falta Jones_matrix
    - Drawing
        - En 5.4.1.2. Several vectors hay [3] un warning  RuntimeWarning: invalid value encountered in sqrt  b = np.sqrt(b2)
        - En Stokes hay un warning  /matplotlib/contour.py:1243: UserWarning: No contour levels were found within the data range.  warnings.warn("No contour levels were found"

* Examples - reordenar ejemplos de fácil a dificil
    - 6.1.2. Malus Law for 2 angles: Mueller formalism ... un poco más de texto
    - 6.1.3. Zeno effect Un poco más de texto
    - 6.2.1.1.2. Purify: Warning: ComplexWarning: Casting complex values to real discards the imaginary part   M[i, j] = elem


* Credits: Logo UCM

* History: 0.2.0

* Pass from Matrix to arrays with @
    In python 3 we can do matrix multiplication using arrays:

        c = a @ b  (in Matlab c = a * b)

        while

        c = a * b  (in Matlab c = a.* b)

        """
        PendingDeprecationWarning: the matrix subclass is not the recommended way to
        represent matrices or deal with linear algebra
        (see https://docs.scipy.org/doc/numpy/user/numpy-for-matlab-users.html).
        Please adjust your code to use regular ndarray.
        """

* Unify the same functions and parameters from Jones-Stokes and Jones-Mueller.
*
