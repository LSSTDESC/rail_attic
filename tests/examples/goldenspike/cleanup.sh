
if [[ -z "${RAILDIR}" ]] then;
   \rm -rf examples/goldenspike/output_*.pq
   \rm -rf examples/goldenspike/output_*.fits
   \rm -rf examples/goldenspike/output_*.hdf5
   \rm -rf examples/goldenspike/*.pkl
   \rm -rf examples/goldenspike/creation_data
   \rm -rf examples/goldenspike/estimation_results
   \rm -rf examples/goldenspike/evaluation_results
   \rm -rf examples/goldenspike/*.out
   \rm -rf examples/goldenspike/data/pretrained_flow_copy.pkl
   \rm -rf examples/goldenspike/tmp_goldenspike.yml
   \rm -rf examples/goldenspike/tmp_goldenspike_config.yml
else
   \rm -rf ${RAILDIR}/examples/goldenspike/output_*.pq
   \rm -rf ${RAILDIR}/examples/goldenspike/output_*.fits
   \rm -rf ${RAILDIR}/examples/goldenspike/output_*.hdf5
   \rm -rf ${RAILDIR}/examples/goldenspike/*.pkl
   \rm -rf ${RAILDIR}/examples/goldenspike/creation_data
   \rm -rf ${RAILDIR}/examples/goldenspike/estimation_results
   \rm -rf ${RAILDIR}/examples/goldenspike/evaluation_results
   \rm -rf ${RAILDIR}/examples/goldenspike/*.out
   \rm -rf ${RAILDIR}/examples/goldenspike/data/pretrained_flow_copy.pkl
   \rm -rf ${RAILDIR}/examples/goldenspike/tmp_goldenspike.yml
   \rm -rf ${RAILDIR}/examples/goldenspike/tmp_goldenspike_config.yml     
fi

