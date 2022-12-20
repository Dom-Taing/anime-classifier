import React, { useState } from "react";
import "./TextBar.css"

let URL = `http://127.0.0.1:5000/`

export default function TextBar() {
    const [input, setInput] = useState("")
    const [result, setResult] = useState([])

    function handleSubmit(e) {
        e.preventDefault();
        fetch(URL, {
            method: 'POST',
            headers: {
            'Accept': 'application/json',
            'Content-Type': 'application/json',
            'Access-Control-Allow-Origin': '*'
            },
            body: JSON.stringify(input)})
            .then((res) => res.json())
            .then((data) => {
                setResult(data)
                console.log(data)
            })
    }

    return <div className="display">
    <div className="Form__Container">
        <form className="Form" onSubmit={handleSubmit}>
            <textarea
            className="Input__Bar"
            placeholder="Type Here"
            value={input}
            onChange={(e) => {setInput(e.target.value)}}
            />
            <input className="button__submit" type="submit"/>
        </form>
    </div>
    <div className="Form__Container">
        <p> It's genres are : {result.map(item => <b>{item}, </b>) }
        </p>
    </div>
</div>
}