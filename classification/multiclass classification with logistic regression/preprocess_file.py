def preprocess(infile, outfile):
    # Original labels
    stext1 = 'Iris-setosa'
    stext2 = 'Iris-versicolor'
    stext3 = 'Iris-virginica'
    # Values for replacement
    rtext1 = '0'
    rtext2 = '1'
    rtext3 = '2'

    fid = open(infile, "r")
    oid = open(outfile, "w")
    
    for string in fid:
        if string.find(stext1)>-1:
            oid.write(string.replace(stext1,rtext1))
        elif string.find(stext2)>-1:
            oid.write(string.replace(stext2,rtext2))
        elif string.find(stext3)>-1:
            oid.write(string.replace(stext3,rtext3))
    fid.close()
    oid.close()

# Preprocessor to remove the test (only needed once)
preprocess('datasets/iris.data','datasets/iris_process.data')
