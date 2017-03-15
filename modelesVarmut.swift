//
//  modelesProto.swift
//  train4ANN
//
//  Created by istros2015 on 18/02/2017.
//  Copyright © 2017 Istros Anlagen sro. All rights reserved.
//

import Foundation

@objc class NN32d : NNd, Varmut {
    
    var metavar : String = "n1Score"
    
    internal convenience init (pop : Bool) {
        self.init(ni:3, nh:2, no:1, ipop : pop)
        self.addplistcoef = "~/Finance/Aetius/dicoA" + nom + suf + ".plist"
        self.addR2data = "~/Finance/Aetius/testR2A" + nom + suf + ".csv"
        self.pop = pop
        ca = 1.0
        cb = 2.0
        decodeNN()
    }
    internal init(ni:NSInteger, nh:NSInteger, no:NSInteger, ipop:Bool) {
        super.init()
        self.nom = "nn32d"
        self.pop = ipop
        self.addplistcoef = "~/Finance/Aetius/dicoA" + nom + suf + ".plist"
        self.addR2data = "~/Finance/Aetius/testR2A" + nom + suf + ".csv"
        // number of input, hidden, and output nodes
        self.ni = ni+1 // +1 for bias node
        self.nh = nh
        self.no = no
        
        ca = 1.0
        cb = 2.0
        
        // activations for nodes
        self.ai = [1.0]*&self.ni
        self.ah = [1.0]*&self.nh
        self.ao = [1.0]*&self.no
        
        //create weights
        self.wi = makeMatrix(self.ni, J: self.nh)
        self.wo = makeMatrix(self.nh, J: self.no)
        
        //    if decodeNN() == false {
        for i in 0...(self.ni-1){
            for j in 0...(self.nh-1){
                self.wi[i][j]=randomFunc(-0.9, b: 0.9)
                //   print(self.wi[i][j])
            }
        }
        
        for j in 0...(self.nh-1){
            for k in 0...(self.no-1){
                self.wo[j][k] = randomFunc(-2.0, b: 2.0)
                //    print(self.wi[j][k])
            }
        }
        //   }
        
        // last change in weights for momentum
        self.ci = makeMatrix(self.ni, J: self.nh)
        self.co = makeMatrix(self.nh, J: self.no)
    }
    /*
    override func determineAddCoefs ()->() {
        if self.pop {
            self.addplistcoef = "~/Finance/Aetius/dicoANN32da.plist"
        } else {
            self.addplistcoef = "~/Finance/Aetius/dicoANN32db.plist"
        }
        decodeNN()
    }
    
     func filteredBase (base : NSArray, nongroupe : String)->(NSArray) {
     let pred : NSPredicate = NSPredicate.init(format:"secteur !=[c] %@", argumentArray:[nongroupe])
     let fb = base.filteredArrayUsingPredicate(pred)
     return fb
     }
     */
    override func train(patterns:Array<Array<Array<Double>>>, iterations:NSInteger, N:Double=0.18, M:Double=0.006){
        // N: learning rate
        // M: momentum factor
        let npatterns = normalise(patterns)
        for i in 0...iterations{
            var error = 0.0
            for p in 0...npatterns.count-1{
                let inputs = npatterns[p][0]
                let targets = npatterns[p][1]
                self.update(inputs)
                error = error + self.backPropagate(targets, N: N, M: M)
                
            }
            // weights()
            if i % 500 == 0{
                print("\(self.nom)-\(self.metavar) error \(error)")
            }
        }
        encodeNN()
        let lr2 = calculeR2(npatterns)
        self.r2 = lr2
        print("\(self.nom) r2:\(lr2)")
        //    print("target pbr:\(targetpbr(npatterns[0][0]))")
    }
    
    override func chargeData(add : String)->(Array<Array<Array<Double>>>) {
        let nadd = (add as NSString).stringByExpandingTildeInPath
        let dat = NSArray(contentsOfFile:nadd)
        let varob = "etafiNY.pbr"
        var arob = Array<Double>()
        for item in dat! {
            let item2  = item as! NSDictionary
            let varob = item2.objectForKey(varob) as! Double
            arob.append(varob)
        }
        let dico = statMono(arob)
        let e = dico["moy"]
        let s = dico["ec"]
        
        var pat = Array<Array<Array<Double>>>()
        if dat!.count >= 90 {
            for item in dat! {
                let item2  = item as! NSDictionary
                let ivarob = item2.objectForKey(varob) as! Double
                let zob = zscore(ivarob, e : e!, s : s!)
                if (ivarob > 0.0)&&(abs(zob) < 3.5) {
                    let roeci : Double = item2.objectForKey("etafiNY.roeci") as! Double
                    let levScore2 = item2.objectForKey("levScore2") as! Double
                    //              let trc4 = item2.objectForKey("trc4") as! Double
                    //         let n1Score = item2.objectForKey("n1Score") as! Double
                    //         let zlatoScore = item2.objectForKey("zlatoScore") as! Double
                    
                    let imetavar = item2.objectForKey(self.metavar) as! Double
                    let itransmeta = transmeta(imetavar)
                    //        let potg = item2.objectForKey("etafiMoy.potg") as! Double
                    //          let roic : Double = item2.objectForKey("etafiNY.roic") as! Double
                    //           let troic = truncsigm(roic,coef: 50.0)
                    //         let fcfyMoy = item2.objectForKey("etafiMoy.fcfy") as! Double
                    //         let tfcfy = truncsigm(fcfyMoy,coef: 40.0)
                    //      let cap = item2.objectForKey("cap") as! Double
                    //      let lcap = logcap(cap)
                    let iar : Array<Array<Double>> = [[roeci, levScore2, itransmeta],[ivarob]]
                    pat.append(iar)
                }
            }
        }
        return pat
    }
    
    override func normaliseInputsBruts(inputs:Array<Double>)->(Array<Double>) {
        let lni = inputs.count
        var ninputs = inputs//[Double](count: lni, repeatedValue:0.0)
        
        for j in 0...lni-1 {
            if j == 4 {
                ninputs[j] = logcap(inputs[j])
            }
            ninputs[j] =  cb * ((ninputs[j] - minInputs[j]) / (maxInputs[j] - minInputs[j])) - ca
        }
        return ninputs
    }
}

@objc class NN42d : NNd, Varmut {
    
    var metavar : String = "n1Score"
    
    internal convenience init (pop : Bool) {
        self.init(ni:4, nh:2, no:1, ipop : pop)
        self.addplistcoef = "~/Finance/Aetius/dicoA" + nom + suf + ".plist"
        self.addR2data = "~/Finance/Aetius/testR2A" + nom + suf + ".csv"
        self.pop = pop
        ca = 1.0
        cb = 2.0
        decodeNN()
    }
    internal init(ni:NSInteger, nh:NSInteger, no:NSInteger, ipop:Bool) {
        super.init()
        self.nom = "nn42d"
        self.pop = ipop
        self.addplistcoef = "~/Finance/Aetius/dicoA" + nom + suf + ".plist"
        self.addR2data = "~/Finance/Aetius/testR2A" + nom + suf + ".csv"
        // number of input, hidden, and output nodes
        self.ni = ni+1 // +1 for bias node
        self.nh = nh
        self.no = no
        
        ca = 1.0
        cb = 2.0
        
        // activations for nodes
        self.ai = [1.0]*&self.ni
        self.ah = [1.0]*&self.nh
        self.ao = [1.0]*&self.no
        
        //create weights
        self.wi = makeMatrix(self.ni, J: self.nh)
        self.wo = makeMatrix(self.nh, J: self.no)
        
        //    if decodeNN() == false {
        for i in 0...(self.ni-1){
            for j in 0...(self.nh-1){
                self.wi[i][j]=randomFunc(-0.9, b: 0.9)
                //   print(self.wi[i][j])
            }
        }
        
        for j in 0...(self.nh-1){
            for k in 0...(self.no-1){
                self.wo[j][k] = randomFunc(-2.0, b: 2.0)
                //    print(self.wi[j][k])
            }
        }
        //   }
        
        // last change in weights for momentum
        self.ci = makeMatrix(self.ni, J: self.nh)
        self.co = makeMatrix(self.nh, J: self.no)
    }
    
    override func train(patterns:Array<Array<Array<Double>>>, iterations:NSInteger, N:Double=0.18, M:Double=0.006){
        // N: learning rate
        // M: momentum factor
        let npatterns = normalise(patterns)
        for i in 0...iterations{
            var error = 0.0
            for p in 0...npatterns.count-1{
                let inputs = npatterns[p][0]
                let targets = npatterns[p][1]
                self.update(inputs)
                error = error + self.backPropagate(targets, N: N, M: M)
                
            }
            // weights()
            if i % 500 == 0{
                print("\(self.nom)-\(self.metavar) error \(error)")
            }
        }
        encodeNN()
        let lr2 = calculeR2(npatterns)
        self.r2 = lr2
        print("\(self.nom)-\(self.metavar) r2:\(lr2)")
        //    print("target pbr:\(targetpbr(npatterns[0][0]))")
    }
    
    override func chargeData(add : String)->(Array<Array<Array<Double>>>) {
        let nadd = (add as NSString).stringByExpandingTildeInPath
        let dat = NSArray(contentsOfFile:nadd)
        let varob = "etafiNY.pbr"
        var arob = Array<Double>()
        for item in dat! {
            let item2  = item as! NSDictionary
            let varob = item2.objectForKey(varob) as! Double
            arob.append(varob)
        }
        let dico = statMono(arob)
        let e = dico["moy"]
        let s = dico["ec"]
        
        var pat = Array<Array<Array<Double>>>()
        if dat!.count >= 90 {
            for item in dat! {
                let item2  = item as! NSDictionary
                let ivarob = item2.objectForKey(varob) as! Double
                let zob = zscore(ivarob, e : e!, s : s!)
                if (ivarob > 0.0)&&(abs(zob) < 3.5) {
                    let roeci : Double = item2.objectForKey("etafiNY.roeci") as! Double
                    let levScore2 = item2.objectForKey("levScore2") as! Double
             //       let trc4 = item2.objectForKey("trc4") as! Double
                    //         let n1Score = item2.objectForKey("n1Score") as! Double
                    //         let zlatoScore = item2.objectForKey("zlatoScore") as! Double
                    
                    let imetavar = item2.objectForKey(self.metavar) as! Double
                    let itransmeta = transmeta(imetavar)
                    //        let potg = item2.objectForKey("etafiMoy.potg") as! Double
                    //          let roic : Double = item2.objectForKey("etafiNY.roic") as! Double
                    //           let troic = truncsigm(roic,coef: 50.0)
                    //         let fcfyMoy = item2.objectForKey("etafiMoy.fcfy") as! Double
                    //         let tfcfy = truncsigm(fcfyMoy,coef: 40.0)
                    let cap = item2.objectForKey("cap") as! Double
                    let lcap = logcap(cap)
                    let iar : Array<Array<Double>> = [[roeci, levScore2, lcap, itransmeta],[ivarob]]
                    pat.append(iar)
                }
            }
        }
        return pat
    }
    
    override func normaliseInputsBruts(inputs:Array<Double>)->(Array<Double>) {
        let lni = inputs.count
        var ninputs = inputs//[Double](count: lni, repeatedValue:0.0)
        
        for j in 0...lni-1 {
            if j == 2 {
                ninputs[j] = logcap(inputs[j])
            }
            ninputs[j] =  cb * ((ninputs[j] - minInputs[j]) / (maxInputs[j] - minInputs[j])) - ca
        }
        return ninputs
    }
}

@objc class NN43d : NNd, Varmut {
    
    var metavar : String = "n1Score"
    
    internal convenience init (pop : Bool) {
        self.init(ni:4, nh:2, no:1, ipop : pop)
        self.addplistcoef = "~/Finance/Aetius/dicoA" + nom + suf + ".plist"
        self.addR2data = "~/Finance/Aetius/testR2A" + nom + suf + ".csv"
        self.pop = pop
        ca = 1.0
        cb = 2.0
        decodeNN()
    }
    internal init(ni:NSInteger, nh:NSInteger, no:NSInteger, ipop:Bool) {
        super.init()
        self.nom = "nn43d"
        self.pop = ipop
        self.addplistcoef = "~/Finance/Aetius/dicoA" + nom + suf + ".plist"
        self.addR2data = "~/Finance/Aetius/testR2A" + nom + suf + ".csv"
        // number of input, hidden, and output nodes
        self.ni = ni+1 // +1 for bias node
        self.nh = nh
        self.no = no
        
        ca = 1.0
        cb = 2.0
        
        // activations for nodes
        self.ai = [1.0]*&self.ni
        self.ah = [1.0]*&self.nh
        self.ao = [1.0]*&self.no
        
        //create weights
        self.wi = makeMatrix(self.ni, J: self.nh)
        self.wo = makeMatrix(self.nh, J: self.no)
        
        //    if decodeNN() == false {
        for i in 0...(self.ni-1){
            for j in 0...(self.nh-1){
                self.wi[i][j]=randomFunc(-0.9, b: 0.9)
                //   print(self.wi[i][j])
            }
        }
        
        for j in 0...(self.nh-1){
            for k in 0...(self.no-1){
                self.wo[j][k] = randomFunc(-2.0, b: 2.0)
                //    print(self.wi[j][k])
            }
        }
        //   }
        
        // last change in weights for momentum
        self.ci = makeMatrix(self.ni, J: self.nh)
        self.co = makeMatrix(self.nh, J: self.no)
    }
    
    override func train(patterns:Array<Array<Array<Double>>>, iterations:NSInteger, N:Double=0.18, M:Double=0.006){
        // N: learning rate
        // M: momentum factor
        let npatterns = normalise(patterns)
        for i in 0...iterations{
            var error = 0.0
            for p in 0...npatterns.count-1{
                let inputs = npatterns[p][0]
                let targets = npatterns[p][1]
                self.update(inputs)
                error = error + self.backPropagate(targets, N: N, M: M)
                
            }
            // weights()
            if i % 500 == 0{
                print("\(self.nom)-\(self.metavar) error \(error)")
            }
        }
        encodeNN()
        let lr2 = calculeR2(npatterns)
        self.r2 = lr2
        print("\(self.nom)-\(self.metavar) r2:\(lr2)")
        //    print("target pbr:\(targetpbr(npatterns[0][0]))")
    }
    
    override func chargeData(add : String)->(Array<Array<Array<Double>>>) {
        let nadd = (add as NSString).stringByExpandingTildeInPath
        let dat = NSArray(contentsOfFile:nadd)
        let varob = "etafiNY.pbr"
        var arob = Array<Double>()
        for item in dat! {
            let item2  = item as! NSDictionary
            let varob = item2.objectForKey(varob) as! Double
            arob.append(varob)
        }
        let dico = statMono(arob)
        let e = dico["moy"]
        let s = dico["ec"]
        
        var pat = Array<Array<Array<Double>>>()
        if dat!.count >= 90 {
            for item in dat! {
                let item2  = item as! NSDictionary
                let ivarob = item2.objectForKey(varob) as! Double
                let zob = zscore(ivarob, e : e!, s : s!)
                if (ivarob > 0.0)&&(abs(zob) < 3.5) {
                    let roeci : Double = item2.objectForKey("etafiNY.roeci") as! Double
                    let levScore2 = item2.objectForKey("levScore2") as! Double
                    //       let trc4 = item2.objectForKey("trc4") as! Double
                    //         let n1Score = item2.objectForKey("n1Score") as! Double
                    //         let zlatoScore = item2.objectForKey("zlatoScore") as! Double
                    
                    let imetavar = item2.objectForKey(self.metavar) as! Double
                    let itransmeta = transmeta(imetavar)
                    //        let potg = item2.objectForKey("etafiMoy.potg") as! Double
                    //          let roic : Double = item2.objectForKey("etafiNY.roic") as! Double
                    //           let troic = truncsigm(roic,coef: 50.0)
                    //         let fcfyMoy = item2.objectForKey("etafiMoy.fcfy") as! Double
                    //         let tfcfy = truncsigm(fcfyMoy,coef: 40.0)
                    let cap = item2.objectForKey("cap") as! Double
                    let lcap = logcap(cap)
                    let iar : Array<Array<Double>> = [[roeci, levScore2, lcap, itransmeta],[ivarob]]
                    pat.append(iar)
                }
            }
        }
        return pat
    }
    
    override func normaliseInputsBruts(inputs:Array<Double>)->(Array<Double>) {
        let lni = inputs.count
        var ninputs = inputs//[Double](count: lni, repeatedValue:0.0)
        
        for j in 0...lni-1 {
            if j == 2 {
                ninputs[j] = logcap(inputs[j])
            }
            ninputs[j] =  cb * ((ninputs[j] - minInputs[j]) / (maxInputs[j] - minInputs[j])) - ca
        }
        return ninputs
    }
}

@objc class NN53d1 : NNd, Varmut {
    
    var metavar : String = "n1Score"
    
    internal convenience init (pop : Bool) {
        self.init(ni:5, nh:3, no:1, ipop : pop)
        self.addplistcoef = "~/Finance/Aetius/dicoA" + nom + suf + ".plist"
        self.addR2data = "~/Finance/Aetius/testR2A" + nom + suf + ".csv"
        self.pop = pop
        ca = 1.0
        cb = 2.0
        decodeNN()
    }
    internal init(ni:NSInteger, nh:NSInteger, no:NSInteger, ipop:Bool) {
        super.init()
        self.nom = "nn53d1"
        self.pop = ipop
        self.addplistcoef = "~/Finance/Aetius/dicoA" + nom + suf + ".plist"
        self.addR2data = "~/Finance/Aetius/testR2A" + nom + suf + ".csv"
        // number of input, hidden, and output nodes
        self.ni = ni+1 // +1 for bias node
        self.nh = nh
        self.no = no
        
        ca = 1.0
        cb = 2.0
        
        // activations for nodes
        self.ai = [1.0]*&self.ni
        self.ah = [1.0]*&self.nh
        self.ao = [1.0]*&self.no
        
        //create weights
        self.wi = makeMatrix(self.ni, J: self.nh)
        self.wo = makeMatrix(self.nh, J: self.no)
        
        //    if decodeNN() == false {
        for i in 0...(self.ni-1){
            for j in 0...(self.nh-1){
                self.wi[i][j]=randomFunc(-0.9, b: 0.9)
                //   print(self.wi[i][j])
            }
        }
        
        for j in 0...(self.nh-1){
            for k in 0...(self.no-1){
                self.wo[j][k] = randomFunc(-2.0, b: 2.0)
                //    print(self.wi[j][k])
            }
        }
        //   }
        
        // last change in weights for momentum
        self.ci = makeMatrix(self.ni, J: self.nh)
        self.co = makeMatrix(self.nh, J: self.no)
    }
    
    override func train(patterns:Array<Array<Array<Double>>>, iterations:NSInteger, N:Double=0.17, M:Double=0.006){
        // N: learning rate
        // M: momentum factor
        let npatterns = normalise(patterns)
        for i in 0...iterations{
            var error = 0.0
            for p in 0...npatterns.count-1{
                let inputs = npatterns[p][0]
                let targets = npatterns[p][1]
                self.update(inputs)
                error = error + self.backPropagate(targets, N: N, M: M)
                
            }
            // weights()
            if i % 1000 == 0{
                print("\(self.nom)-\(self.metavar) error \(error)")
            }
        }
        encodeNN()
        let lr2 = calculeR2(npatterns)
        self.r2 = lr2
        print("\(self.nom)-\(self.metavar) r2:\(lr2)")
        //    print("target pbr:\(targetpbr(npatterns[0][0]))")
    }
    
    override func chargeData(add : String)->(Array<Array<Array<Double>>>) {
        let nadd = (add as NSString).stringByExpandingTildeInPath
        let dat = NSArray(contentsOfFile:nadd)
        let varob = "etafiNY.pbr"
        var arob = Array<Double>()
        for item in dat! {
            let item2  = item as! NSDictionary
            let varob = item2.objectForKey(varob) as! Double
            arob.append(varob)
        }
        let dico = statMono(arob)
        let e = dico["moy"]
        let s = dico["ec"]
        
        var pat = Array<Array<Array<Double>>>()
        if dat!.count >= 90 {
            for item in dat! {
                let item2  = item as! NSDictionary
                let ivarob = item2.objectForKey(varob) as! Double
                let zob = zscore(ivarob, e : e!, s : s!)
                if (ivarob > 0.0)&&(abs(zob) < 3.5) {
                    let roeci : Double = item2.objectForKey("etafiNY.roeci") as! Double
                    let levScore2 = item2.objectForKey("levScore2") as! Double
                    let epsvar = item2.objectForKey("etafiVar.eps") as! Double
                    let tepsvar = truncsigm(epsvar,coef: 40.0)
                    //         let n1Score = item2.objectForKey("n1Score") as! Double
                    //         let zlatoScore = item2.objectForKey("zlatoScore") as! Double
                    
                    let imetavar = item2.objectForKey(self.metavar) as! Double
                    let itransmeta = transmeta(imetavar)
                    //        let potg = item2.objectForKey("etafiMoy.potg") as! Double
                    //          let roic : Double = item2.objectForKey("etafiNY.roic") as! Double
                    //           let troic = truncsigm(roic,coef: 50.0)
                    //         let fcfyMoy = item2.objectForKey("etafiMoy.fcfy") as! Double
                    //         let tfcfy = truncsigm(fcfyMoy,coef: 40.0)
                    let cap = item2.objectForKey("cap") as! Double
                    let lcap = logcap(cap)
                    let iar : Array<Array<Double>> = [[roeci, levScore2, tepsvar, itransmeta, lcap],[ivarob]]
                    pat.append(iar)
                }
            }
        }
        return pat
    }
    
    override func normaliseInputsBruts(inputs:Array<Double>)->(Array<Double>) {
        let lni = inputs.count
        var ninputs = inputs//[Double](count: lni, repeatedValue:0.0)
        
        for j in 0...lni-1 {
            if j == 4 {
                ninputs[j] = logcap(inputs[j])
            }
            ninputs[j] =  cb * ((ninputs[j] - minInputs[j]) / (maxInputs[j] - minInputs[j])) - ca
        }
        return ninputs
    }
}

@objc class NN42g : NN, Varmut {
    
    var metavar : String = "n1Score"
    
    internal convenience override init () {
        self.init(ni:4, nh:2, no:1)
        self.addplistcoef = "~/Finance/Aetius/dicoA" + nom + ".plist"
        self.addR2data = "~/Finance/Aetius/testR2A" + nom + ".csv"
        ca = 1.0
        cb = 2.0
        decodeNN()
    }
    internal init(ni:NSInteger, nh:NSInteger, no:NSInteger) {
        super.init()
        self.nom = "nn42g"
        self.addplistcoef = "~/Finance/Aetius/dicoA" + nom + ".plist"
        self.addR2data = "~/Finance/Aetius/testR2A" + nom + ".csv"
        // number of input, hidden, and output nodes
        self.ni = ni+1 // +1 for bias node
        self.nh = nh
        self.no = no
        
        ca = 1.0
        cb = 2.0
        
        // activations for nodes
        self.ai = [1.0]*&self.ni
        self.ah = [1.0]*&self.nh
        self.ao = [1.0]*&self.no
        
        //create weights
        self.wi = makeMatrix(self.ni, J: self.nh)
        self.wo = makeMatrix(self.nh, J: self.no)
        
        //    if decodeNN() == false {
        for i in 0...(self.ni-1){
            for j in 0...(self.nh-1){
                self.wi[i][j]=randomFunc(-0.9, b: 0.9)
                //   print(self.wi[i][j])
            }
        }
        
        for j in 0...(self.nh-1){
            for k in 0...(self.no-1){
                self.wo[j][k] = randomFunc(-1.0, b: 1.0)
                //    print(self.wi[j][k])
            }
        }
        //   }
        
        // last change in weights for momentum
        self.ci = makeMatrix(self.ni, J: self.nh)
        self.co = makeMatrix(self.nh, J: self.no)
    }
    
    override func train(patterns:Array<Array<Array<Double>>>, iterations:NSInteger, N:Double=0.18, M:Double=0.006){
        // N: learning rate
        // M: momentum factor
        let npatterns = normalise(patterns)
        for i in 0...iterations{
            var error = 0.0
            for p in 0...npatterns.count-1{
                let inputs = npatterns[p][0]
                let targets = npatterns[p][1]
                self.update(inputs)
                error = error + self.backPropagate(targets, N: N, M: M)
                
            }
            // weights()
            if i % 700 == 0{
                print("\(self.nom)-\(self.metavar) error \(error)")
            }
        }
        encodeNN()
        let lr2 = calculeR2(npatterns)
        self.r2 = lr2
        print("\(self.nom)-\(self.metavar) r2:\(lr2)")
        //    print("target pbr:\(targetpbr(npatterns[0][0]))")
    }
    
    override func chargeData(add : String)->(Array<Array<Array<Double>>>) {
        let nadd = (add as NSString).stringByExpandingTildeInPath
        let dat = NSArray(contentsOfFile:nadd)
        let varob = "etafiNY.pbr"
        var arob = Array<Double>()
        for item in dat! {
            let item2  = item as! NSDictionary
            let varob = item2.objectForKey(varob) as! Double
            arob.append(varob)
        }
        let dico = statMono(arob)
        let e = dico["moy"]
        let s = dico["ec"]
        
        var pat = Array<Array<Array<Double>>>()
        if dat!.count >= 90 {
            for item in dat! {
                let item2  = item as! NSDictionary
                let ivarob = item2.objectForKey(varob) as! Double
                let zob = zscore(ivarob, e : e!, s : s!)
                if (ivarob > 0.0)&&(abs(zob) < 3.5) {
                    let roeci : Double = item2.objectForKey("etafiNY.roeci") as! Double
                    let levScore2 = item2.objectForKey("levScore2") as! Double
                    //       let trc4 = item2.objectForKey("trc4") as! Double
                    //         let n1Score = item2.objectForKey("n1Score") as! Double
                    //         let zlatoScore = item2.objectForKey("zlatoScore") as! Double
                    
                    let imetavar = item2.objectForKey(self.metavar) as! Double
                    let itransmeta = transmeta(imetavar)
                    //        let potg = item2.objectForKey("etafiMoy.potg") as! Double
                    //          let roic : Double = item2.objectForKey("etafiNY.roic") as! Double
                    //           let troic = truncsigm(roic,coef: 50.0)
                    //         let fcfyMoy = item2.objectForKey("etafiMoy.fcfy") as! Double
                    //         let tfcfy = truncsigm(fcfyMoy,coef: 40.0)
                    let cap = item2.objectForKey("cap") as! Double
                    let lcap = logcap(cap)
                    let iar : Array<Array<Double>> = [[roeci, levScore2, lcap, itransmeta],[ivarob]]
                    pat.append(iar)
                }
            }
        }
        return pat
    }
    
    override func normaliseInputsBruts(inputs:Array<Double>)->(Array<Double>) {
        let lni = inputs.count
        var ninputs = inputs//[Double](count: lni, repeatedValue:0.0)
        
        for j in 0...lni-1 {
            if j == 2 {
                ninputs[j] = logcap(inputs[j])
            }
            ninputs[j] =  cb * ((ninputs[j] - minInputs[j]) / (maxInputs[j] - minInputs[j])) - ca
        }
        return ninputs
    }
}

@objc class NN53g : NN, Varmut {
    
    var metavar : String = "n1Score"
    
    internal convenience override init () {
        self.init(ni:5, nh:3, no:1)
        self.addplistcoef = "~/Finance/Aetius/dicoA" + nom + ".plist"
        self.addR2data = "~/Finance/Aetius/testR2A" + nom + ".csv"
        ca = 1.0
        cb = 2.0
        decodeNN()
    }
    internal init(ni:NSInteger, nh:NSInteger, no:NSInteger) {
        super.init()
        self.nom = "nn53g"
        self.addplistcoef = "~/Finance/Aetius/dicoA" + nom + ".plist"
        self.addR2data = "~/Finance/Aetius/testR2A" + nom + ".csv"
        // number of input, hidden, and output nodes
        self.ni = ni+1 // +1 for bias node
        self.nh = nh
        self.no = no
        
        ca = 1.0
        cb = 2.0
        
        // activations for nodes
        self.ai = [1.0]*&self.ni
        self.ah = [1.0]*&self.nh
        self.ao = [1.0]*&self.no
        
        //create weights
        self.wi = makeMatrix(self.ni, J: self.nh)
        self.wo = makeMatrix(self.nh, J: self.no)
        
        //    if decodeNN() == false {
        for i in 0...(self.ni-1){
            for j in 0...(self.nh-1){
                self.wi[i][j]=randomFunc(-0.9, b: 0.9)
                //   print(self.wi[i][j])
            }
        }
        
        for j in 0...(self.nh-1){
            for k in 0...(self.no-1){
                self.wo[j][k] = randomFunc(-1.0, b: 1.0)
                //    print(self.wi[j][k])
            }
        }
        //   }
        
        // last change in weights for momentum
        self.ci = makeMatrix(self.ni, J: self.nh)
        self.co = makeMatrix(self.nh, J: self.no)
    }
    
    override func train(patterns:Array<Array<Array<Double>>>, iterations:NSInteger, N:Double=0.18, M:Double=0.006){
        // N: learning rate
        // M: momentum factor
        let npatterns = normalise(patterns)
        for i in 0...iterations{
            var error = 0.0
            for p in 0...npatterns.count-1{
                let inputs = npatterns[p][0]
                let targets = npatterns[p][1]
                self.update(inputs)
                error = error + self.backPropagate(targets, N: N, M: M)
                
            }
            // weights()
            if i % 700 == 0{
                print("\(self.nom)-\(self.metavar) error \(error)")
            }
        }
        encodeNN()
        let lr2 = calculeR2(npatterns)
        self.r2 = lr2
        print("\(self.nom)-\(self.metavar) r2:\(lr2)")
        //    print("target pbr:\(targetpbr(npatterns[0][0]))")
    }
    
    override func chargeData(add : String)->(Array<Array<Array<Double>>>) {
        let nadd = (add as NSString).stringByExpandingTildeInPath
        let dat = NSArray(contentsOfFile:nadd)
        let varob = "etafiNY.pbr"
        var arob = Array<Double>()
        for item in dat! {
            let item2  = item as! NSDictionary
            let varob = item2.objectForKey(varob) as! Double
            arob.append(varob)
        }
        let dico = statMono(arob)
        let e = dico["moy"]
        let s = dico["ec"]
        
        var pat = Array<Array<Array<Double>>>()
        if dat!.count >= 90 {
            for item in dat! {
                let item2  = item as! NSDictionary
                let ivarob = item2.objectForKey(varob) as! Double
                let zob = zscore(ivarob, e : e!, s : s!)
                if (ivarob > 0.0)&&(abs(zob) < 3.5) {
                    let roeci : Double = item2.objectForKey("etafiNY.roeci") as! Double
                    let levScore2 = item2.objectForKey("levScore2") as! Double
                    //       let trc4 = item2.objectForKey("trc4") as! Double
                    let n2Score = item2.objectForKey("n2Score") as! Double
                    //         let zlatoScore = item2.objectForKey("zlatoScore") as! Double
                    
                    let imetavar = item2.objectForKey(self.metavar) as! Double
                    let itransmeta = transmeta(imetavar)
                    //        let potg = item2.objectForKey("etafiMoy.potg") as! Double
                    //          let roic : Double = item2.objectForKey("etafiNY.roic") as! Double
                    //           let troic = truncsigm(roic,coef: 50.0)
                    //         let fcfyMoy = item2.objectForKey("etafiMoy.fcfy") as! Double
                    //         let tfcfy = truncsigm(fcfyMoy,coef: 40.0)
                    let cap = item2.objectForKey("cap") as! Double
                    let lcap = logcap(cap)
                    let iar : Array<Array<Double>> = [[roeci, levScore2, n2Score, lcap, itransmeta],[ivarob]]
                    pat.append(iar)
                }
            }
        }
        return pat
    }
    
    override func normaliseInputsBruts(inputs:Array<Double>)->(Array<Double>) {
        let lni = inputs.count
        var ninputs = inputs//[Double](count: lni, repeatedValue:0.0)
        
        for j in 0...lni-1 {
            if j == 2 {
                ninputs[j] = logcap(inputs[j])
            }
            ninputs[j] =  cb * ((ninputs[j] - minInputs[j]) / (maxInputs[j] - minInputs[j])) - ca
        }
        return ninputs
    }
}

@objc class NN63kg : NNb, Varmut {
    
    var metavar : String = "n1Score"
    
    internal convenience override init () {
        self.init(ni:6, nh:3, no:1)
        self.addplistcoef = "~/Finance/Aetius/dicoA" + nom + ".plist"
        self.addR2data = "~/Finance/Aetius/testR2A" + nom + ".csv"
        ca = 1.0
        cb = 2.0
        decodeNN()
    }
    internal init(ni:NSInteger, nh:NSInteger, no:NSInteger) {
        super.init()
        self.nom = "nn63kg"
        self.addplistcoef = "~/Finance/Aetius/dicoA" + nom + ".plist"
        self.addR2data = "~/Finance/Aetius/testR2A" + nom + ".csv"
        // number of input, hidden, and output nodes
        self.ni = ni+1 // +1 for bias node
        self.nh = nh
        self.no = no
        
        ca = 1.0
        cb = 2.0
        
        // activations for nodes
        self.ai = [1.0]*&self.ni
        self.ah = [1.0]*&self.nh
        self.ao = [1.0]*&self.no
        
        //create weights
        self.wi = makeMatrix(self.ni, J: self.nh)
        self.wo = makeMatrix(self.nh, J: self.no)
        
        //    if decodeNN() == false {
        for i in 0...(self.ni-1){
            for j in 0...(self.nh-1){
                self.wi[i][j]=randomFunc(-0.9, b: 0.9)
                //   print(self.wi[i][j])
            }
        }
        
        for j in 0...(self.nh-1){
            for k in 0...(self.no-1){
                self.wo[j][k] = randomFunc(-2.0, b: 2.0)
                //    print(self.wi[j][k])
            }
        }
        //   }
        
        // last change in weights for momentum
        self.ci = makeMatrix(self.ni, J: self.nh)
        self.co = makeMatrix(self.nh, J: self.no)
    }
    
    override func train(patterns:Array<Array<Array<Double>>>, iterations:NSInteger, N:Double=0.19, M:Double=0.008){
        // N: learning rate
        // M: momentum factor
        let npatterns = normalise(patterns)
        for i in 0...iterations{
            var error = 0.0
            for p in 0...npatterns.count-1{
                let inputs = npatterns[p][0]
                let targets = npatterns[p][1]
                self.update(inputs)
                error = error + self.backPropagate(targets, N: N, M: M)
                
            }
            // weights()
            if i % 700 == 0{
                print("\(self.nom)-\(self.metavar) error \(error)")
            }
        }
        encodeNN()
        let lr2 = calculeR2(npatterns)
        self.r2 = lr2
        print("\(self.nom)-\(self.metavar) r2:\(lr2)")
        //    print("target pbr:\(targetpbr(npatterns[0][0]))")
    }
    
    override func chargeData(add : String)->(Array<Array<Array<Double>>>) {
        let nadd = (add as NSString).stringByExpandingTildeInPath
        let dat = NSArray(contentsOfFile:nadd)
        let varob = "etafiNY.pbr"
        /*
        var arob = Array<Double>()
        for item in dat! {
            let item2  = item as! NSDictionary
            let varob = item2.objectForKey(varob) as! Double
            arob.append(varob)
        }
        let dico = statMono(arob)
        let e = dico["moy"]
        let s = dico["ec"]
        */
        var pat = Array<Array<Array<Double>>>()
        if dat!.count >= 90 {
            for item in dat! {
                let item2  = item as! NSDictionary
                let ivarob = item2.objectForKey(varob) as! Double
           //     let zob = zscore(ivarob, e : e!, s : s!)
                if (ivarob > 0.0) {
                    let roeci : Double = item2.objectForKey("etafiNY.roeci") as! Double
                    let levScore2 = item2.objectForKey("levScore2") as! Double
                    //       let trc4 = item2.objectForKey("trc4") as! Double
                    let n2Score = item2.objectForKey("n2Score") as! Double
                    //         let zlatoScore = item2.objectForKey("zlatoScore") as! Double
                    
                    let imetavar = item2.objectForKey(self.metavar) as! Double
                    let itransmeta = transmeta(imetavar)
                    //        let potg = item2.objectForKey("etafiMoy.potg") as! Double
                    //          let roic : Double = item2.objectForKey("etafiNY.roic") as! Double
                    //           let troic = truncsigm(roic,coef: 50.0)
                    let fcfyMoy = item2.objectForKey("etafiMoy.fcfy") as! Double
                    let tfcfy = truncsigm(fcfyMoy,coef: 40.0)
                    let cap = item2.objectForKey("cap") as! Double
                    let lcap = logcap(cap)
                    let iar : Array<Array<Double>> = [[roeci, levScore2, n2Score, tfcfy, lcap, itransmeta],[ivarob]]
                    pat.append(iar)
                }
            }
        }
        return pat
    }
    
    override func normaliseInputsBruts(inputs:Array<Double>)->(Array<Double>) {
        let lni = inputs.count
        var ninputs = inputs//[Double](count: lni, repeatedValue:0.0)
        
        for j in 0...lni-1 {
            if j == 2 {
                ninputs[j] = logcap(inputs[j])
            }
            ninputs[j] =  cb * ((ninputs[j] - minInputs[j]) / (maxInputs[j] - minInputs[j])) - ca
        }
        return ninputs
    }
}

@objc class NN63g : NN, Varmut {
    
    var metavar : String = "n1Score"
    
    internal convenience override init () {
        self.init(ni:6, nh:3, no:1)
        self.addplistcoef = "~/Finance/Aetius/dicoA" + nom + ".plist"
        self.addR2data = "~/Finance/Aetius/testR2A" + nom + ".csv"
        ca = 1.0
        cb = 2.0
        decodeNN()
    }
    internal init(ni:NSInteger, nh:NSInteger, no:NSInteger) {
        super.init()
        self.nom = "nn63g"
        self.addplistcoef = "~/Finance/Aetius/dicoA" + nom + ".plist"
        self.addR2data = "~/Finance/Aetius/testR2A" + nom + ".csv"
        // number of input, hidden, and output nodes
        self.ni = ni+1 // +1 for bias node
        self.nh = nh
        self.no = no
        
        ca = 1.0
        cb = 2.0
        
        // activations for nodes
        self.ai = [1.0]*&self.ni
        self.ah = [1.0]*&self.nh
        self.ao = [1.0]*&self.no
        
        //create weights
        self.wi = makeMatrix(self.ni, J: self.nh)
        self.wo = makeMatrix(self.nh, J: self.no)
        
        //    if decodeNN() == false {
        for i in 0...(self.ni-1){
            for j in 0...(self.nh-1){
                self.wi[i][j]=randomFunc(-0.9, b: 0.9)
                //   print(self.wi[i][j])
            }
        }
        
        for j in 0...(self.nh-1){
            for k in 0...(self.no-1){
                self.wo[j][k] = randomFunc(-2.0, b: 2.0)
                //    print(self.wi[j][k])
            }
        }
        //   }
        
        // last change in weights for momentum
        self.ci = makeMatrix(self.ni, J: self.nh)
        self.co = makeMatrix(self.nh, J: self.no)
    }
    
    override func train(patterns:Array<Array<Array<Double>>>, iterations:NSInteger, N:Double=0.19, M:Double=0.008){
        // N: learning rate
        // M: momentum factor
        let npatterns = normalise(patterns)
        for i in 0...iterations{
            var error = 0.0
            for p in 0...npatterns.count-1{
                let inputs = npatterns[p][0]
                let targets = npatterns[p][1]
                self.update(inputs)
                error = error + self.backPropagate(targets, N: N, M: M)
                
            }
            // weights()
            if i % 700 == 0{
                print("\(self.nom)-\(self.metavar) error \(error)")
            }
        }
        encodeNN()
        let lr2 = calculeR2(npatterns)
        self.r2 = lr2
        print("\(self.nom)-\(self.metavar) r2:\(lr2)")
        //    print("target pbr:\(targetpbr(npatterns[0][0]))")
    }
    
    override func chargeData(add : String)->(Array<Array<Array<Double>>>) {
        let nadd = (add as NSString).stringByExpandingTildeInPath
        let dat = NSArray(contentsOfFile:nadd)
        let varob = "etafiNY.pbr"
        var arob = Array<Double>()
        for item in dat! {
            let item2  = item as! NSDictionary
            let varob = item2.objectForKey(varob) as! Double
            arob.append(varob)
        }
        let dico = statMono(arob)
        let e = dico["moy"]
        let s = dico["ec"]
        
        var pat = Array<Array<Array<Double>>>()
        if dat!.count >= 90 {
            for item in dat! {
                let item2  = item as! NSDictionary
                let ivarob = item2.objectForKey(varob) as! Double
                let zob = zscore(ivarob, e : e!, s : s!)
                if (ivarob > 0.0)&&(abs(zob) < 3.5) {
                    let roeci : Double = item2.objectForKey("etafiNY.roeci") as! Double
                    let levScore2 = item2.objectForKey("levScore2") as! Double
                    //       let trc4 = item2.objectForKey("trc4") as! Double
                    let n2Score = item2.objectForKey("n2Score") as! Double
                    //         let zlatoScore = item2.objectForKey("zlatoScore") as! Double
                    
                    let imetavar = item2.objectForKey(self.metavar) as! Double
                    let itransmeta = transmeta(imetavar)
                    //        let potg = item2.objectForKey("etafiMoy.potg") as! Double
                    //          let roic : Double = item2.objectForKey("etafiNY.roic") as! Double
                    //           let troic = truncsigm(roic,coef: 50.0)
                    let fcfyMoy = item2.objectForKey("etafiMoy.fcfy") as! Double
                    let tfcfy = truncsigm(fcfyMoy,coef: 40.0)
                    let cap = item2.objectForKey("cap") as! Double
                    let lcap = logcap(cap)
                    let iar : Array<Array<Double>> = [[roeci, levScore2, n2Score, tfcfy, lcap, itransmeta],[ivarob]]
                    pat.append(iar)
                }
            }
        }
        return pat
    }
    
    override func normaliseInputsBruts(inputs:Array<Double>)->(Array<Double>) {
        let lni = inputs.count
        var ninputs = inputs//[Double](count: lni, repeatedValue:0.0)
        
        for j in 0...lni-1 {
            if j == 2 {
                ninputs[j] = logcap(inputs[j])
            }
            ninputs[j] =  cb * ((ninputs[j] - minInputs[j]) / (maxInputs[j] - minInputs[j])) - ca
        }
        return ninputs
    }
}
