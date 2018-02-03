import chainer
from chainer import functions as F
from chainer import links as L
from chainer import \
     cuda, gradient_check, optimizers, serializers, utils, \
     Chain, ChainList, Function, Link, Variable

#onehot関数の定義
#n個の0の中からxビット目を1にする
def onehot(x, n):
    ret = np.zeros(n).astype(np.float32)
    ret[x] = 1.0
    return ret
##astypeは、ndarrayの要素のデータ型を別のデータ型にしたndarrayを生成する
##要素を変更しても元のndarrayには影響しない
##ndarrayはnumpyの配列クラス

##overlap関数の定義
##結局のところ、何やってる？
def overlap(u, v): #u, v: (1 * -) Variable -> (1 * 1) Variable
    denominator = F.sqrt(F.batch_l2_norm_squared(u) * F.batch_l2_norm_squared(v))
    if(np.array_equal(denominator.data, np.array([0]))):
        return F.matmul(u, F.transpose(v))
    return F.matmul(u, F.transpose(v)) / F.reshape(denominator,(1,1))
##denominator = 分母
##batch_l2_norm_squared(x)は、ベクトル上にL2ノルム(=大きさ)の2乗を実装する。
##てことはF.sqrt=の部分が出しているのは2乗平均？
##np.array_equal(x,y)は、各配列xとyを比較して等しければtrue,そうでなければfalseを返す。
##今回はdenominatorが0ベクトル行列ならF.matmulを実行
##matmulは2つの配列の行列乗算を計算する。
##transpose(v)は入力変数の寸法を固定する(2つめの引数が無い場合次元を逆にする)
##reshapeは入力変数の形状を変更する(reshape(a,(3,2))なら配列aを3*2行列にしてくれる感じ)。

##C関数の定義
##コンテンツルックアップ演算：ヘッドの操作をやってるみたい(Method1ページ目の右下あたりの式)
def C(M, k, beta):
    # (N * W), (1 * W), (1 * 1) -> (N * 1)
    # (not (N * W), ({R,1} * W), (1 * {R,1}) -> (N * {R,1}))
    W = M.data.shape[1]
    ret_list = [0] * M.data.shape[0]
    for i in range(M.data.shape[0]):
        #M.data.shape行列の0番目に入っている数だけ処理
        ret_list[i] = overlap(F.reshape(M[i,:], (1, W)), k) * beta # 第i行を選ぶ
    return F.transpose(F.softmax(F.transpose(F.concat(ret_list, 0)))) #垂直方向に連結する
    #そして各列のsoftmaxを計算する
    #△△.○○という表現は、△△の中の○○という意味合いで使われる。
    ##ここでは、どっかで定義しているMの中のdetaを使っている。
    ##M.data.shapeは何？
    ##concat = 複数の文字列をひとつの文字列に集め、連結させること。指定された変数を軸に連結する。
    ##M[i,:]は、文字列Mの中のi番目から最後までの文字列を取り出す

##u2a関数の定義
##使用率ベクトルを使って割り当て重みを算出する計算をしていると思う
def u2a(u): #u, a: (N * 1) Variable
    N = len(u.data)
    #↓メモリロケーションのインデックスを昇順にソートしたものがφ
    phi = np.argsort(u.data.reshape(N)) #u.data[phi]: ascending(上昇)
    a_list = [0] * N
    cumprod = Variable(np.array([[1.0]]).astype(np.float32))
    for i in range(N):
        #N(len(u.data))回繰り返し
        a_list[phi[i]] = cumprod * (1.0 - F.reshape(u[phi[i],0], (1,1)))
        cumprod *= F.reshape(u[phi[i],0], (1,1))
    return F.concat(a_list, 0) #垂直方向に連結する
##phiは多分φ(ファイ)。cumprodは累積積の意味。
##argsortは昇順にソートしたインデックス(何番目の要素か)の配列を返す。



class DeepLSTM(Chain):
    def __init__(self, d_in, d_out):
        super(DeepLSTM, self).__init__(
            l1 = L.LSTM(d_in, d_out),
            l2 = L.Linear(d_out, d_out),)
    def __call__(self, x):
        self.x = x
        self.y = self.l2(self.l1(self.x))
        return self.y
    def reset_state(self):
        self.l1.reset_state()
##LSTM：チェーンとして完全に接続されたLSTM層。子リンクとして上向きおよび横向きの接続を保持する。
##前のタイムステップでセルの状態や出力などの状態を保持する。よってステートフルLSTMとして使用できる
##Linear(全結合層): liniear()関数をラップするリンク
#superclassのメソッドを呼び出す際はsuper(自クラス, インスタンス)のように記述
#super(DeepLSTM, self).__init__(処理):super()を使って、superclassの__init__()を呼び出す


class DNC(Chain):
    def __init__(self, X, Y, N, W, R):
        self.X = X #入力の次元
        self.Y = Y #出力の次元
        self.N = N #メモリ数
        self.W = W #メモリ列の1行あたりの次元
        self.R = R #リードヘッドの数
        self.controller = DeepLSTM(W*R+X, Y+W*R+3*W+5*R+3)

        super(DNC, self).__init__(
            l_dl = self.controller,
            l_wr = L.Linear(self.R * self.W, self.Y)
            )
        self.reset_state()
    def __call__(self, x):
        #concat; self.r次元について、split_size個の配列になるよう分割
        self.chi = F.concat((x, self.r))
        ##↓call呼び出し
        (self.nu, self.xi) = \
                  F.split_axis(self.l_dl(self.chi), [self.Y], 1)  #(8)
        (self.kr, self.betar, self.kw, self.betaw,
         self.e, self.v, self.f, self.ga, self.gw, self.pi
         ) = F.split_axis(self.xi, np.cumsum(
             [self.W*self.R, self.R, self.W, 1, self.W, self.W, self.R, 1, 1]), 1) #(10)

        self.kr = F.reshape(self.kr, (self.R, self.W)) # R * W
        self.betar = 1 + F.softplus(self.betar) # 1 * R
        # self.kw: 1 * W
        self.betaw = 1 + F.softplus(self.betaw) # 1 * 1
        self.e = F.sigmoid(self.e) # 1 * W
        # self.v : 1 * W
        self.f = F.sigmoid(self.f) # 1 * R
        self.ga = F.sigmoid(self.ga) # 1 * 1
        self.gw = F.sigmoid(self.gw) # 1 * 1
        self.pi = F.softmax(F.reshape(self.pi, (self.R, 3))) # R * 3 (softmax for 3)

        # self.wr : N * R
        self.psi_mat = 1 - F.matmul(Variable(np.ones((self.N, 1)).astype(np.float32)), self.f) * self.wr # N * R
        self.psi = Variable(np.ones((self.N, 1)).astype(np.float32)) # N * 1
        for i in range(self.R):
            self.psi = self.psi * F.reshape(self.psi_mat[:,i],(self.N,1)) # N * 1

        # self.ww, self.u : N * 1
        self.u = (self.u + self.ww - (self.u * self.ww)) * self.psi

        self.a = u2a(self.u) # N * 1
        self.cw = C(self.M, self.kw, self.betaw) # N * 1
        #↓書き込み重み(Methodの2ページ目、左カラム真ん中の式)
        self.ww = F.matmul(F.matmul(self.a, self.ga) + F.matmul(self.cw, 1.0 - self.ga), self.gw) # N * 1

        #↓(メモリへの)書き込みベクトルの式
        self.M = self.M * (np.ones((self.N, self.W)).astype(np.float32) - F.matmul(self.ww, self.e)) + F.matmul(self.ww, self.v) # N * W

        self.p = (1.0 - F.matmul(Variable(np.ones((self.N,1)).astype(np.float32)), F.reshape(F.sum(self.ww),(1,1)))) \
                  * self.p + self.ww # N * 1
        self.wwrep = F.matmul(self.ww, Variable(np.ones((1, self.N)).astype(np.float32))) # N * N
        self.L = (1.0 - self.wwrep - F.transpose(self.wwrep)) * self.L + F.matmul(self.ww, F.transpose(self.p)) # N * N
        self.L = self.L * (np.ones((self.N, self.N)) - np.eye(self.N)) # force L[i,i] == 0

        self.fo = F.matmul(self.L, self.wr) # N * R
        self.ba = F.matmul(F.transpose(self.L), self.wr) # N * R

        self.cr_list = [0] * self.R
        for i in range(self.R):
            self.cr_list[i] = C(self.M, F.reshape(self.kr[i,:],(1, self.W)),
                                F.reshape(self.betar[0,i],(1, 1))) # N * 1
        self.cr = F.concat(self.cr_list) # N * R

        self.bacrfo = F.concat((F.reshape(F.transpose(self.ba),(self.R,self.N,1)),
                               F.reshape(F.transpose(self.cr),(self.R,self.N,1)),
                               F.reshape(F.transpose(self.fo) ,(self.R,self.N,1)),),2) # R * N * 3
        self.pi = F.reshape(self.pi, (self.R,3,1)) # R * 3 * 1
        self.wr = F.transpose(F.reshape(F.batch_matmul(self.bacrfo, self.pi), (self.R, self.N))) # N * R ## N*R行列に直している

        ##self.r ... Mとwrの行列乗算を1*RWの行列に直したもの
        self.r = F.reshape(F.matmul(F.transpose(self.M), self.wr),(1, self.R * self.W)) # W * R (-> 1 * RW)

        self.y = self.l_Wr(self.r) + self.nu # 1 * Y (9)
        return self.y

        ##リセット
        def reset_state(self):
            self.l_dl.reset_state()
            self.u = Variable(np.zeros((self.N, 1)).astype(np.float32))
            self.p = Variable(np.zeros((self.N, 1)).astype(np.float32))
            self.L = Variable(np.zeros((self.N, self.N)).astype(np.float32))
            self.M = Variable(np.zeros((self.N, self.W)).astype(np.float32))
            self.r = Variable(np.zeros((1, self.R*self.W)).astype(np.float32))
            self.wr = Variable(np.zeros((self.N, self.R)).astype(np.float32))
            self.ww = Variable(np.zeros((self.N, 1)).astype(np.float32))
            # any variable else ?

##各パラメータを設定
    X = 5
    Y = 5
    N = 10
    W = 10
    R = 2
    #前向き演算開始
    mdl = DNC(X, Y, N, W, R)
    opt = optimizers.Adam()
    opt.setup(mdl)
    #datanum = 100000
    datanum = 1000
    loss = 0.0
    acc = 0.0
    for datacnt in range(datanum):
        lossfrac = np.zeros((1,2))
        # x_seq = np.random.rand(X,seqlen).astype(np.float32)
        # t_seq = np.random.rand(Y,seqlen).astype(np.float32)
        # t_seq = np.copy(x_seq)

        contentlen = np.random.randint(3,6)
        content = np.random.randint(0,X-1,contentlen)
        seqlen = contentlen + contentlen
        x_seq_list = [float('nan')] * seqlen
        t_seq_list = [float('nan')] * seqlen
        for i in range(seqlen):
            if (i < contentlen):
                x_seq_list[i] = onehot(content[i],X)
            elif (i == contentlen):
                x_seq_list[i] = onehot(X-1,X)
            else:
                x_seq_list[i] = np.zeros(X).astype(np.float32)

            if (i >= contentlen):
                t_seq_list[i] = onehot(content[i-contentlen],X)

        mdl.reset_state()
        for cnt in range(seqlen):
            x = Variable(x_seq_list[cnt].reshape(1,X))
            if (isinstance(t_seq_list[cnt], np.ndarray)):
                t = Variable(t_seq_list[cnt].reshape(1,Y))
            else:
                t = []

            y = mdl(x)
            if (isinstance(t,chainer.Variable)):
                loss += (y - t)**2
                print y.data, t.data, np.argmax(y.data)==np.argmax(t.data)
                if (np.argmax(y.data)==np.argmax(t.data)): acc += 1
            if (cnt+1==seqlen):
                mdl.cleargrads()
                loss.grad = np.ones(loss.data.shape, dtype=np.float32)
                loss.backward()
                opt.update()
                loss.unchain_backward()
                print '(', datacnt, ')', loss.data.sum()/loss.data.size/contentlen, acc/contentlen
                lossfrac += [loss.data.sum()/loss.data.size/seqlen, 1.]
                loss = 0.0
                acc = 0.0

#(self.kr, self.betar, self.kw, self.betaw, self.e, self.v, self.f, self.ga, self.gw, self.pi) =
# F.split_axis(self.xi, np.cumsum([self.W*self.R, self.R, self.W, 1, self.W, self.W, self.R, 1, 1]), 1)
 ##配列xiを、軸に沿ってnp.cumsum()個の等しい配列に分割
 ##分割される軸...1
