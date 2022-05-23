PYTHONPATH=$(which py)

if [[ -z "$PYHTONPATH" ]]
then read -p "Please enter your python (.exe) path manually: " PYTHONPATH
else read -p "Is $PYTHONPATH the path for your preferred python version? If no, please enter your python path manually. (y/n): " ans

    if [[ $ans != "y" ]]
    then read -p "Please enter your python (.exe) path manually: " PYTHONPATH
    fi

fi

if [[ $* != "" ]]
then $PYTHONPATH ./pipeline.py $*
else $PYTHONPATH ./pipeline.py -h
fi
