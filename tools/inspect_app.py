from streamlit.testing.v1 import AppTest

app = AppTest.from_file('app.py', default_timeout=180)
app.run(timeout=180)
print('Exception:', app.exception)
print('Multiselect count:', len(app.multiselect))
if app.multiselect:
    sel = app.multiselect[0]
    print('Options sample (first 20):')
    print(sel.options[:20])
    print('Total options:', len(sel.options))
    targets = ['Kaşıntı','Deri Döküntüsü','Bulantı']
    present = [t for t in targets if t in sel.options]
    print('Targets present:', present)
