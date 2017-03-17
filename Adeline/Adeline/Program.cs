using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Adeline
{
    abstract class NeuronBase
    {
        private static Random random = new Random();
        protected List<double> weights;
        protected Func<double, double> ActivationFunction { get; set; }
        public abstract void Train(double eps, List<List<double>> trainInput, List<double> trainOutput);

        public NeuronBase(int nInputs)
        {
            weights = Enumerable.Repeat(0.0, nInputs).ToList();
        }
        protected void InitWeights(double startRange, double endRange)
        {
            for (int i = 0; i < weights.Count; i++)
            {
                weights[i] =
                    random.NextDouble() * (endRange - startRange)
                     + startRange;
            }
        }
        public double CalculateOutput(List<double> inputs)
        {

            return ActivationFunction(GetLinearPart(inputs));
        }

        public double GetLinearPart(List<double> inputs)
        {
            return inputs.Zip(weights, (x, w) => x * w)
                .Sum();
        }
    }

    class SigmoidNeuron : NeuronBase
    {
        private double SigmoidDerivate(double x)
        {
            double h = 1e-5;
            return (ActivationFunction(x + h) - ActivationFunction(x + h)) / (2 * h);
            //return (-2.0 * beta * Math.Exp(beta * x) * Math.Exp(-beta * x) - beta * Math.Exp(beta * x) + beta * Math.Exp(-beta * x)) /
            //     (Math.Pow(1 + Math.Exp(-beta * x), 2.0));

            //return (beta * (1 - Math.Exp(beta * x)) / (Math.Exp(beta * x) * Math.Pow(1.0 / Math.Exp(beta * x), 2)) - (beta * Math.Exp(beta * x)) / (1.0 / (Math.Exp(beta * x) + 1)));
        }
        private double _eta, beta =5.0;
        public SigmoidNeuron(int nInputs, double eta) : base(nInputs)
        
        {
            ActivationFunction = x => Math.Tanh(beta * x);
            _eta = eta;
        }
                
        public override void Train(double eps, List<List<double>> trainInput, List<double> trainOutput)
        {
            InitWeights(-1, 1);
            double n=0;
            int error = 0;
            int epoka = 0;
            do
            {
                int indeks = 0;
                foreach (var input in trainInput)
                {
                    var routput = trainOutput[indeks];
                    var output = CalculateOutput(input);
                    var s = GetLinearPart(input);

                        for (int i = 0; i < weights.Count; i++)
                        {
                            weights[i] += _eta*(routput- output)*SigmoidDerivate(s,5.0)*input[i];
                        }
                    indeks++;

                }
                epoka++;
                error = 0;
                indeks = 0;
                foreach (var input in trainInput)
                {
                    var routput = trainOutput[indeks];
                    var output = CalculateOutput(input);

                    if (Math.Abs(routput - output) > 1e-10)
                    {
                        for (int i = 0; i < weights.Count; i++)
                        {
                            error++;
                        }
                    }
                    indeks++;
                }
                Console.WriteLine($"EPOKA:{epoka}, BŁĄD:{error}");
            } while (error > eps);
        }

        class Program
        {
            static void Main(string[] args)
            {
                var inputs = new List<List<double>>
                {
                    new List<double> {1,2,1 },
                    new List<double> {1,2,2 },
                    new List<double> {1,0,6 },
                    new List<double> {1,-2,10 },
                    new List<double> {1,-2,0 },
                    new List<double> {1,0,0 },
                    new List<double> {1,4,-20}
                };

                var outputs = new List<double>
            {
                1,1,1,-1,-1,-1,-1
            };
                var adaline = new SigmoidNeuron(3,0.8);

                adaline.Train(0, inputs, outputs);
                int index = 0;
                foreach (var input in inputs)
                {
                    var output = adaline.CalculateOutput(input);
                    var required = outputs[index];
                    Console.WriteLine($"{index}:{output} = {required}");
                    index++;
                }
            }
        }
    }
}
