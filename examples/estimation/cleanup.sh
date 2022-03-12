
if [[ -z "${RAILDIR}" ]] then;
   \rm -rf examples/estimation/TEMPZFILE.out
   \rm -rf examples/estimation/output_*.fits
else
   \rm -rf ${RAILDIR}/examples/estimation/TEMPZFILE.out
   \rm -rf ${RAILDIR}/examples/estimation/output_*.fits
fi

