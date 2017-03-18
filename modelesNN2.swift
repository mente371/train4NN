//
//  modelesNN2.swift
//  train4ANN
//
//  Created by istros2015 on 14/03/2017.
//  Copyright © 2017 Istros Anlagen sro. All rights reserved.
//

import Foundation

@objc class NNb : NN {
    
    var ee = Array<Double>()  // moy de la var target
    var ss = Array<Double>()  // ecart-type de la var target
    
    override func normalise (patterns:Array<Array<Array<Double>>>)->(Array<Array<Array<Double>>>) {
        majMaxMins(patterns)
        let ni = patterns[0][0].count
        var maxminInputs = [Double](count: ni, repeatedValue:0.0)
        for j in 0...ni-1 {
            maxminInputs[j] = maxInputs[j] - minInputs[j]
        }
        let no = patterns[0][1].count
        var maxminOutputs = [Double](count: no, repeatedValue:0.0)
        for j in 0...no-1 {
            maxminOutputs[j] = maxOutputs[j] - minOutputs[j]
        }
        
        var npatterns = Array<Array<Array<Double>>>()
        ee = [Double](count: patterns[0][1].count, repeatedValue:0.0)
        ss = [Double](count: patterns[0][1].count, repeatedValue:1.0)
        
        let itar = patterns.map { $0[1]}
        for p in 0...itar[0].count-1 {
            let itarp = itar.map { $0[p]}
            let dico = statMono(itarp)
            let e = dico["moy"]
            let s = dico["ec"]
            
                if let ei = e {
                    ee[p] = ei
                }
                if let si = s {
                    ss[p] = si
                }
        }
        
        for p in 0...patterns.count-1{
            var inputs = patterns[p][0]
            var targets = patterns[p][1]
            
            for j in 0...inputs.count-1 {
                inputs[j] =  cb * ((inputs[j] - minInputs[j]) / maxminInputs[j]) - ca
            }
            for j in 0...targets.count-1 {
                //    targets[j] =  cb * ((targets[j] - minOutputs[j]) / maxminOutputs[j]) - ca
                targets[j] = logsigmoid(targets[j], e: ee[j], s: ss[j])
            }
            npatterns.append([inputs,targets])
        }
        
        return npatterns
    }
    
    func logsigmoid (x:Double, e: Double, s: Double)->(Double) {
        return logistic(zscore(x, e: e, s: s))
    }
    
    func denormlog(x:Double, e:Double, s:Double) -> (Double) {
        //   return min + ((x + 1.0) / 2.0 ) * (max - min)
        //      return min + (x * (max - min))
       // return (max - min) * ((x + ca)/cb) + min
        return e-s*log((1.0/x)-1.0)
    }
    
    override func targetpbr(inputs : Array<Double>)->(Array<Double>) {
        var aro = Array<Double>()
        for k in 0...no-1 {
            let mod = self.update(inputs)[k]
            let modn = denormlog(mod, e: ee[k], s: ss[k])
            aro.append(modn)
        }
        return aro
    }
    
    override func calculeR2 (patterns:Array<Array<Array<Double>>>)->(Double) {
        var arin = Array<Array<Double>>()
        for p in 0...patterns.count-1{
            let inputs = patterns[p][0]
            let targets = patterns[p][1]
            for k in 0...no-1 {
                let mod = self.update(inputs)[k]
                let modn = denormlog(mod, e: ee[k], s: ss[k])
                let reel = targets[k]
                let reeln = denormlog(reel, e: ee[k], s: ss[k])
                let ari = [modn, reeln]
                arin.append(ari)
            }
        }
        var r2 : Double = 0.0
        var fXi : Double = 0.0
        var fYi : Double = 0.0
        var SSres : Double = 0.0
        var SStot : Double = 0.0
        var fSX : Double = 0.0
        var fSY : Double = 0.0
        var fSX2 : Double = 0.0
        var fSY2 : Double = 0.0
        var fSXY : Double = 0.0
        var fYti : Double = 0.0
        var eY : Double = 0.0
        var h : UInt32 = 0
        var num : Double = 0.0
        var denom : Double = 0.0
        var corr : Double = 0.0
        var newar = Array<Array<Double>>()
        for i in 0...arin.count-1 {
            fXi = arin[i][0] // modn
            fYi = arin[i][1] // reeln
            let testX : Bool = (fXi > -99999.99) && (fXi < 99999.99)
            let testY : Bool = (fYi > -99999.99) && (fYi < 99999.99)
            let test : Bool = testX && testY
            if test {
                SSres += pow((fYi - fXi),2.0)
                fSY += fYi
                h += 1
                newar.append([fXi, fYi])
                fSX += fXi
                fSXY += (fXi * fYi)
                fSX2 += (fXi * fXi)
                fSY2 += (fYi * fYi)
            }
        }
        let dh = Double(h)
        if h != 0 {
            eY = fSY / dh
            num = fSXY - (fSX * fSY)/dh;
            denom = sqrt((fSX2-(fSX*fSX)/dh)*(fSY2-(fSY*fSY)/dh));
        }
        
        if (denom != 0.0) {
            corr = num / denom;
        }
        for j in 0...newar.count-1 {
            fYti = newar[j][1] - eY
            SStot += pow(fYti,2.0)
        }
        encodeR2data(newar)
        statModele(newar)
        if SStot > 0.0 {
            r2 = 1.0 - (SSres / SStot)
        }
        self.r2 = r2
        self.corr = corr
        self.npop = h
        self.sr = self.calculSR(corr, pop: h)
        return r2
    }
    
    override func calculeR2array (patterns:Array<Array<Array<Double>>>)->(Array<Double>) {
        var arin = Array<Array<Double>>()
        for p in 0...patterns.count-1{
            let inputs = patterns[p][0]
            let targets = patterns[p][1]
            for k in 0...0 {  //no-1
                let mod = self.update(inputs)[k]
                let modn = denormlog(mod, e: ee[k], s: ss[k])
                let reel = targets[k]
                let reeln = denormlog(reel, e: ee[k], s: ss[k])
                let ari = [modn, reeln]
                arin.append(ari)
            }
        }
        var r2 : Double = 0.0
        var fXi : Double = 0.0
        var fYi : Double = 0.0
        var SSres : Double = 0.0
        var SStot : Double = 0.0
        var fSX : Double = 0.0
        var fSY : Double = 0.0
        var fSX2 : Double = 0.0
        var fSY2 : Double = 0.0
        var fSXY : Double = 0.0
        var fYti : Double = 0.0
        var eY : Double = 0.0
        var h : UInt32 = 0
        var npos : UInt = 0
        var nposb : UInt = 0
        var num : Double = 0.0
        var denom : Double = 0.0
        var corr : Double = 0.0
        var newar = Array<Array<Double>>()
        for i in 0...arin.count-1 {
            fXi = arin[i][0] // modn
            fYi = arin[i][1] // reeln
            let np : UInt = fXi > 0.01 ? 1 : 0
            npos += np
            let npb : UInt = (fXi > 0.01)&&(fYi > 0.01) ? 1 : 0
            nposb += npb
            let testX : Bool = (fXi > -99999.99) && (fXi < 99999.99)
            let testY : Bool = (fYi > -99999.99) && (fYi < 99999.99)
            let test : Bool = testX && testY
            if test {
                SSres += pow((fYi - fXi),2.0)
                fSY += fYi
                h += 1
                newar.append([fXi, fYi])
                fSX += fXi
                fSXY += (fXi * fYi)
                fSX2 += (fXi * fXi)
                fSY2 += (fYi * fYi)
            }
        }
        let dh = Double(h)
        if h != 0 {
            eY = fSY / dh
            num = fSXY - (fSX * fSY)/dh;
            denom = sqrt((fSX2-(fSX*fSX)/dh)*(fSY2-(fSY*fSY)/dh));
        }
        if (denom != 0.0) {
            corr = num / denom;
        }
        for j in 0...newar.count-1 {
            fYti = newar[j][1] - eY
            SStot += pow(fYti,2.0)
        }
        if SStot > 0.0 {
            r2 = 1.0 - (SSres / SStot)
        }
        // encodeR2Test
        var text : String = String()
        text.appendContentsOf("model, reel\n")
        for i in 0...newar.count-1 {
            let ari = newar[i]
            text.appendContentsOf("\(ari[0]), \(ari[1])\n")
        }
        let addr2Test = "~/Finance/Aetius/r2TestnnX.csv"
        let nadd = (addr2Test as NSString).stringByExpandingTildeInPath
        let text2 = text as NSString
        try! text2.writeToFile(nadd, atomically: true, encoding: NSUnicodeStringEncoding)
        // fin encodeR2Test
        //  self.npos = npos
        //  self.nposb = nposb
        let lsr = self.calculSR(corr, pop: h)
        return [r2, corr, Double(h), lsr]
    }
}

@objc class NN53k : NNb {
    
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
        self.nom = "NN53k"
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
                self.wi[i][j]=randomFunc(-0.6, b: 0.6)
                //   print(self.wi[i][j])
            }
        }
        
        for j in 0...(self.nh-1){
            for k in 0...(self.no-1){
                self.wo[j][k] = randomFunc(-0.8, b: 0.8)
                //    print(self.wi[j][k])
            }
        }
        //   }
        
        // last change in weights for momentum
        self.ci = makeMatrix(self.ni, J: self.nh)
        self.co = makeMatrix(self.nh, J: self.no)
        
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
        if dat!.count >= 100 {
            for item in dat! {
                let item2  = item as! NSDictionary
                let ivarob = item2.objectForKey(varob) as! Double
          //      let zob = zscore(ivarob, e : e!, s : s!)
                if (ivarob > 0.0) {
                    let roeci : Double = item2.objectForKey("etafiNY.roeci") as! Double
                    let levScore2 = item2.objectForKey("levScore2") as! Double
                    //            let zlatoScore = item2.objectForKey("zlatoScore") as! Double
                    let n2Score = item2.objectForKey("n2Score") as! Double
                    //      let implGeva = item2.objectForKey("implGeva") as! Double
                    //     let trc4 = item2.objectForKey("trc4") as! Double
                    let fcfyMoy = item2.objectForKey("etafiMoy.fcfy") as! Double
                    let tfcfy = truncsigm(fcfyMoy,coef: 30.0)
                    //         let potg : Double = item2.objectForKey("etafiMoy.potg") as! Double
                    //        let tpotg = truncsigm(potg,coef: 30.0)
                    //             let varroce = item2.objectForKey("etafiVar.roce") as! Double
                    //             let tvarroce = truncsigm(varroce,coef: 20.0)
                    let cap = item2.objectForKey("cap") as! Double
                    let lcap = logcap(cap)
                    let iar : Array<Array<Double>> = [[roeci, levScore2, n2Score, tfcfy, lcap],[ivarob]]
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
            if j == 3 {
                ninputs[j] = truncsigm(inputs[j], coef:30.0)
            }
            if j == 4 {
                ninputs[j] = logcap(inputs[j])
            }
            ninputs[j] = cb * ((ninputs[j] - minInputs[j]) / (maxInputs[j] - minInputs[j])) - ca
        }
        return ninputs
    }
}

@objc class NN63k : NNb {
    
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
        self.nom = "NN63k"
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
                self.wi[i][j]=randomFunc(-0.6, b: 0.6)
                //   print(self.wi[i][j])
            }
        }
        
        for j in 0...(self.nh-1){
            for k in 0...(self.no-1){
                self.wo[j][k] = randomFunc(-0.8, b: 0.8)
                //    print(self.wi[j][k])
            }
        }
        //   }
        
        // last change in weights for momentum
        self.ci = makeMatrix(self.ni, J: self.nh)
        self.co = makeMatrix(self.nh, J: self.no)
        
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
        if dat!.count >= 100 {
            for item in dat! {
                let item2  = item as! NSDictionary
                let ivarob = item2.objectForKey(varob) as! Double
                //      let zob = zscore(ivarob, e : e!, s : s!)
                if (ivarob > 0.0) {
                    let roeci : Double = item2.objectForKey("etafiNY.roeci") as! Double
                    let levScore2 = item2.objectForKey("levScore2") as! Double
                    //            let zlatoScore = item2.objectForKey("zlatoScore") as! Double
                    let n2Score = item2.objectForKey("n2Score") as! Double
                    //      let implGeva = item2.objectForKey("implGeva") as! Double
                    //     let trc4 = item2.objectForKey("trc4") as! Double
                    let fcfyMoy = item2.objectForKey("etafiMoy.fcfy") as! Double
                    let tfcfy = truncsigm(fcfyMoy,coef: 30.0)
                    //         let potg : Double = item2.objectForKey("etafiMoy.potg") as! Double
                    //        let tpotg = truncsigm(potg,coef: 30.0)
                    //             let varroce = item2.objectForKey("etafiVar.roce") as! Double
                    //             let tvarroce = truncsigm(varroce,coef: 20.0)
                    let vareva = item2.objectForKey("etafiVar.eva") as! Double
                    let tvareva = truncsigm(vareva,coef: 40.0)
                    let cap = item2.objectForKey("cap") as! Double
                    let lcap = logcap(cap)
                    let iar : Array<Array<Double>> = [[roeci, levScore2, n2Score, tfcfy, tvareva, lcap],[ivarob]]
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
            if j == 3 {
                ninputs[j] = truncsigm(inputs[j], coef:30.0)
            }
            if j == 4 {
                ninputs[j] = truncsigm(inputs[j], coef:40.0)
            }
            if j == 5 {
                ninputs[j] = logcap(inputs[j])
            }
            ninputs[j] = cb * ((ninputs[j] - minInputs[j]) / (maxInputs[j] - minInputs[j])) - ca
        }
        return ninputs
    }
}

@objc class NN73k : NNb {
    
    internal convenience override init () {
        self.init(ni:7, nh:3, no:1)
        self.addplistcoef = "~/Finance/Aetius/dicoA" + nom + ".plist"
        self.addR2data = "~/Finance/Aetius/testR2A" + nom + ".csv"
        ca = 1.0
        cb = 2.0
        decodeNN()
    }
    internal init(ni:NSInteger, nh:NSInteger, no:NSInteger) {
        super.init()
        self.nom = "NN73k"
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
                self.wi[i][j]=randomFunc(-0.6, b: 0.6)
                //   print(self.wi[i][j])
            }
        }
        
        for j in 0...(self.nh-1){
            for k in 0...(self.no-1){
                self.wo[j][k] = randomFunc(-0.8, b: 0.8)
                //    print(self.wi[j][k])
            }
        }
        //   }
        
        // last change in weights for momentum
        self.ci = makeMatrix(self.ni, J: self.nh)
        self.co = makeMatrix(self.nh, J: self.no)
        
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
        if dat!.count >= 100 {
            for item in dat! {
                let item2  = item as! NSDictionary
                let ivarob = item2.objectForKey(varob) as! Double
                //      let zob = zscore(ivarob, e : e!, s : s!)
                if (ivarob > 0.0) {
                    let roeci : Double = item2.objectForKey("etafiNY.roeci") as! Double
                    let levScore2 = item2.objectForKey("levScore2") as! Double
                    //            let zlatoScore = item2.objectForKey("zlatoScore") as! Double
                    let n2Score = item2.objectForKey("n2Score") as! Double
                    //      let implGeva = item2.objectForKey("implGeva") as! Double
                    //     let trc4 = item2.objectForKey("trc4") as! Double
                    let fcfyMoy = item2.objectForKey("etafiMoy.fcfy") as! Double
                    let tfcfy = truncsigm(fcfyMoy,coef: 30.0)
                    let potg : Double = item2.objectForKey("etafiMoy.potg") as! Double
                    let tpotg = truncsigm(potg,coef: 30.0)
                    //             let varroce = item2.objectForKey("etafiVar.roce") as! Double
                    //             let tvarroce = truncsigm(varroce,coef: 20.0)
                    let vareva = item2.objectForKey("etafiVar.eva") as! Double
                    let tvareva = truncsigm(vareva,coef: 40.0)
                    let cap = item2.objectForKey("cap") as! Double
                    let lcap = logcap(cap)
                    let iar : Array<Array<Double>> = [[roeci, levScore2, n2Score, tfcfy, tvareva, tpotg, lcap],[ivarob]]
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
            if j == 3 {
                ninputs[j] = truncsigm(inputs[j], coef:30.0)
            }
            if j == 4 {
                ninputs[j] = truncsigm(inputs[j], coef:40.0)
            }
            if j == 5 {
                ninputs[j] = truncsigm(inputs[j], coef:30.0)
            }
            if j == 6 {
                ninputs[j] = logcap(inputs[j])
            }
            ninputs[j] = cb * ((ninputs[j] - minInputs[j]) / (maxInputs[j] - minInputs[j])) - ca
        }
        return ninputs
    }
}

@objc class NN73ka : NNb {
    
    internal convenience override init () {
        self.init(ni:7, nh:3, no:1)
        self.addplistcoef = "~/Finance/Aetius/dicoA" + nom + ".plist"
        self.addR2data = "~/Finance/Aetius/testR2A" + nom + ".csv"
        ca = 1.0
        cb = 2.0
        decodeNN()
    }
    internal init(ni:NSInteger, nh:NSInteger, no:NSInteger) {
        super.init()
        self.nom = "nn73ka"
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
                self.wi[i][j]=randomFunc(-0.6, b: 0.6)
                //   print(self.wi[i][j])
            }
        }
        
        for j in 0...(self.nh-1){
            for k in 0...(self.no-1){
                self.wo[j][k] = randomFunc(-0.8, b: 0.8)
                //    print(self.wi[j][k])
            }
        }
        //   }
        
        // last change in weights for momentum
        self.ci = makeMatrix(self.ni, J: self.nh)
        self.co = makeMatrix(self.nh, J: self.no)
        
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
        if dat!.count >= 100 {
            for item in dat! {
                let item2  = item as! NSDictionary
                let ivarob = item2.objectForKey(varob) as! Double
                //      let zob = zscore(ivarob, e : e!, s : s!)
                if (ivarob > 0.0) {
                    let roeci : Double = item2.objectForKey("etafiNY.roeci") as! Double
                    let levScore2 = item2.objectForKey("levScore2") as! Double
                    //            let zlatoScore = item2.objectForKey("zlatoScore") as! Double
                    let n3Score = item2.objectForKey("n3Score") as! Double
                    //      let implGeva = item2.objectForKey("implGeva") as! Double
                    //     let trc4 = item2.objectForKey("trc4") as! Double
                    let fcfyMoy = item2.objectForKey("etafiMoy.fcfy") as! Double
                    let tfcfy = truncsigm(fcfyMoy,coef: 30.0)
            //        let potg : Double = item2.objectForKey("etafiMoy.potg") as! Double
            //        let tpotg = truncsigm(potg,coef: 30.0)
                    //             let varroce = item2.objectForKey("etafiVar.roce") as! Double
                    //             let tvarroce = truncsigm(varroce,coef: 20.0)
                    let vareps = item2.objectForKey("etafiVar.eps") as! Double
                    let tvareps = truncsigm(vareps,coef: 40.0)
                    let vareva = item2.objectForKey("etafiVar.eva") as! Double
                    let tvareva = truncsigm(vareva,coef: 40.0)
                    let cap = item2.objectForKey("cap") as! Double
                    let lcap = logcap(cap)
                    let iar : Array<Array<Double>> = [[roeci, levScore2, n3Score, tfcfy, tvareva, tvareps, lcap],[ivarob]]
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
            if j == 3 {
                ninputs[j] = truncsigm(inputs[j], coef:30.0)
            }
            if j == 4 {
                ninputs[j] = truncsigm(inputs[j], coef:40.0)
            }
            if j == 5 {
                ninputs[j] = truncsigm(inputs[j], coef:40.0)
            }
            if j == 6 {
                ninputs[j] = logcap(inputs[j])
            }
            ninputs[j] = cb * ((ninputs[j] - minInputs[j]) / (maxInputs[j] - minInputs[j])) - ca
        }
        return ninputs
    }
}