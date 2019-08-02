#!/bin/bash

out_model_path=$1 # Change this to your folder name.

echo "to_lite...."
cd ${out_model_path}
tar -cvf "../${out_model_path}.tar" "./"
cd ..

model_tool to_lite "${out_model_path}.tar" "${out_model_path}.tar2"

echo "encrypt...."
model_tool encrypt "${out_model_path}.tar2" "${out_model_path}.model"

echo "Done. File 'encrypt_${out_model_path}' created."

rm "${out_model_path}.tar" 
rm "${out_model_path}.tar2"
