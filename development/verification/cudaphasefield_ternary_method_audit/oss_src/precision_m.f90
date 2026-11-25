module precision_m
    integer, parameter :: singlePrecision = kind(0.0)
    integer, parameter :: doublePrecision = kind(0.0d0)
    !integer, parameter :: fp_kind = singlePrecision
	integer, parameter :: fp_kind = doublePrecision

end module precision_m
