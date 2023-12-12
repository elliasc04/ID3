import ID3, parse, random
def testPruningOnHouseData(inFile):
    withPruning = []
    withoutPruning = []

    data = parse.parse(inFile)
    for r in range(10, 310, 10):
        pruningSum = 0
        noPruningSum = 0
        for i in range(100):
            random.shuffle(data)
            train = data[:r]
            valid = data[r:r+ r//4]
            test = data[r+r//4:]
          
            tree = ID3.ID3(train, 'democrat')
          
            ID3.prune(tree, valid)
            acc = ID3.test(tree, test)
            print("pruned tree test accuracy: ",acc)
            pruningSum += acc



            tree = ID3.ID3(train, 'democrat')

            acc = ID3.test(tree, test)
            print("test accuracy: ",acc)
            noPruningSum += acc

        avgPruning = pruningSum / 100
        avgNoPruning = noPruningSum / 100
        withPruning.append(avgPruning)
        withoutPruning.append(avgNoPruning)
            
        print(avgPruning)
        print(avgNoPruning)
        print("average with pruning",sum(withPruning)/len(withPruning)," without: ",sum(withoutPruning)/len(withoutPruning))

    outFile = open("test_result.txt", "w")
    outFile.write(str(withPruning))
    outFile.write("\n\n")
    outFile.write(str(withoutPruning))
    outFile.close()

testPruningOnHouseData('house_votes_84.data')
        

